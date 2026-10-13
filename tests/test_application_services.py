from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json
import socket
import subprocess
import sys
import numpy as np
import pytest
import requests

from configs.runtime_config import load_app_config, PROJECT_ROOT
from configs import network_config
from src.core.classification import EyeDiseaseDetector, DISEASES
from src.core.history import HistoryDB
from src.core.llm import DeepSeekAPI


class FakeModel:
    task = "classify"
    names = dict(enumerate(("A", "C", "D", "G", "H", "M", "N", "O")))

    def __init__(self, probabilities=None):
        values = probabilities if probabilities is not None else [0.02] * 6 + [0.86, 0.02]
        self.results = [SimpleNamespace(probs=SimpleNamespace(data=np.array(values)))]

    def predict(self, image, **kwargs):
        return self.results


def detector_for(tmp_path, model=None):
    path = tmp_path / "fixture.pt"
    path.write_bytes(b"injected test backend, not a real checkpoint")
    detector = EyeDiseaseDetector(model_factory=lambda _: model or FakeModel())
    assert detector.load_model(path)
    return detector


def test_missing_weights_and_failed_switch_clear_old_model(tmp_path):
    detector = detector_for(tmp_path)
    assert not detector.load_model(tmp_path / "missing.pt")
    assert detector.model is None
    assert not detector.classify(np.zeros((12, 12, 3), dtype=np.uint8)).success


def test_nonclassification_weight_rejected(tmp_path):
    model = FakeModel()
    model.task = "detect"
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"test")
    detector = EyeDiseaseDetector(model_factory=lambda _: model)
    assert not detector.load_model(path)


@pytest.mark.parametrize("image", [None, np.array([], dtype=np.uint8),
    np.zeros((4, 4, 2), dtype=np.uint8), np.zeros((4, 4, 3), dtype=float)])
def test_invalid_images_fail_without_a_disease(tmp_path, image):
    result = detector_for(tmp_path).classify(image)
    assert not result.success and result.disease_name is None and result.confidence == 0


@pytest.mark.parametrize("probabilities", [[0] * 8, [float("nan")] * 8, [0.5] * 8, [1.0] * 7])
def test_invalid_probabilities_never_become_default_amd(tmp_path, probabilities):
    result = detector_for(tmp_path, FakeModel(probabilities)).classify(np.zeros((8, 8, 3), dtype=np.uint8))
    assert not result.success and result.disease_name is None


def test_class_mapping_comes_from_weight_names(tmp_path):
    model = FakeModel([0.86] + [0.02] * 7)
    model.names = dict(enumerate(("N", "A", "C", "D", "G", "H", "M", "O")))
    result = detector_for(tmp_path, model).classify(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.success and result.disease_name == "Normal"
    assert set(result.probabilities) == set(DISEASES)


def test_batch_keeps_file_correspondence_and_continues_after_read_failure(tmp_path):
    import cv2
    good = tmp_path / "valid.png"
    assert cv2.imwrite(str(good), np.zeros((8, 8, 3), dtype=np.uint8))
    missing = tmp_path / "missing.png"
    outputs = list(detector_for(tmp_path).classify_files([missing, good]))
    assert outputs[0][0] == str(missing) and not outputs[0][1].success
    assert outputs[1][0] == str(good) and outputs[1][1].success


def test_valid_classification_can_be_persisted_and_queried(tmp_path):
    result = detector_for(tmp_path).classify(np.zeros((8, 8, 3), dtype=np.uint8))
    history = HistoryDB(tmp_path / "history.db")
    history.add("r1", "2026-10-12 12:00:00", "fixture.png", result.disease_name, result.confidence)
    history.update_advice("r1", "online test response")
    record = HistoryDB(tmp_path / "history.db").get_all()[0]
    assert record["disease_name"] == "Normal" and record["advice"] == "online test response"
    history.delete_by_record_id("r1")
    assert history.count() == 0


@pytest.mark.parametrize("disease,confidence", [("未知", 0.98), (None, 0.98), ("AMD", float("nan")), ("AMD", 1.2)])
def test_invalid_records_not_saved(tmp_path, disease, confidence):
    history = HistoryDB(tmp_path / "history.db")
    with pytest.raises(ValueError):
        history.add("r1", "now", "fixture", disease, confidence)
    assert history.count() == 0


def test_api_disconnect_is_bounded_and_labelled(tmp_path):
    calls = []
    def offline(*args, **kwargs):
        calls.append(kwargs)
        raise requests.ConnectionError("test offline")
    api = DeepSeekAPI("test-key", post=offline, sleep=lambda _: None)
    text = api.get_custom_advice("测试问题")
    assert not api.last_response.success and api.last_response.source == "error"
    assert "不可用" in text and len(calls) == api.config.api_attempts
    assert calls[0]["timeout"] == (5, 30)
    result = api.request_treatment_advice("Normal", 0.8)
    assert not result.success and result.source == "local_reference"
    assert "非DeepSeek在线回答" in result.content


def test_invalid_classification_does_not_call_api():
    calls = []
    api = DeepSeekAPI("test-key", post=lambda *a, **k: calls.append(k))
    with pytest.raises(ValueError):
        api.get_treatment_advice("未知", 0.98)
    assert not calls


def test_chat_remembers_only_bounded_successful_turns():
    payloads = []
    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "测试回答"}}]})
    api = DeepSeekAPI("test-key", config=replace(load_app_config(), chat_turns=2), post=post)
    for question in ("问题一", "问题二", "问题三"):
        api.get_custom_advice(question)
    assert payloads[1]["messages"][1]["content"] == "问题一"
    assert len(api._history) == 4 and api._history[0]["content"] == "问题二"


def test_bad_api_response_and_auth_failure_are_explicit():
    for status, body in [(200, {"choices": []}), (401, {})]:
        api = DeepSeekAPI("test-key", post=lambda *a, **k: SimpleNamespace(status_code=status, json=lambda: body))
        api.get_custom_advice("测试问题")
        assert not api.last_response.success


def test_lan_discovery_excludes_proxy_virtual_and_disconnected_interfaces():
    entries = {name: [SimpleNamespace(family=socket.AF_INET, address=ip)] for name, ip in
        [("Meta", "198.18.0.1"), ("VMware VMnet8", "192.168.102.1"),
         ("WLAN", "192.168.1.101"), ("以太网", "192.168.5.2") ]}
    stats = {name: SimpleNamespace(isup=name != "以太网") for name in entries}
    candidates = network_config.get_ipv4_interfaces(entries, stats)
    assert candidates == [{"interface": "WLAN", "ip": "192.168.1.101", "score": 2}]


def test_network_manual_save_and_auto_selection(tmp_path, monkeypatch):
    path = tmp_path / "system.json"
    path.write_text(json.dumps({"network": {}, "startup": {}, "services": {"main_system": True}}))
    network_config.save_network_config("192.168.1.22", False, path=path)
    monkeypatch.setattr(network_config, "SYSTEM_CONFIG_PATH", path)
    assert network_config.get_local_ip() == "192.168.1.22"
    network_config.save_network_config("192.168.1.22", True, "WLAN", path)
    monkeypatch.setattr(network_config, "get_ipv4_interfaces", lambda: [{"interface": "WLAN", "ip": "192.168.1.101"}])
    assert network_config.get_local_ip() == "192.168.1.101"
    assert json.loads(path.read_text())["services"]["main_system"]


def test_pc_diagnosis_server_import_does_not_load_qt():
    result = subprocess.run([sys.executable, "-c",
        "from src.pc.pc_diagnosis_server import DiagnosisEngine; import sys; assert 'PyQt5' not in sys.modules; print('headless OK')"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr


def test_history_pages_and_real_statistics(tmp_path):
    from src.core.history import HistoryDB
    db = HistoryDB(tmp_path / "pages.db")
    for i in range(205):
        db.add(str(i), "2026-10-13 12:00:00", "image.png", "Normal", 0.8)
    db.add("demo", "2026-10-13 12:00:00", "demo.png", "[演示]AMD", 0.9)
    first, total, page = db.get_page(0)
    second, _, _ = db.get_page(1)
    last, _, page = db.get_page(99)
    assert total == 206 and len(first) == len(second) == 100
    assert len(last) == 6 and page == 2
    assert not ({r["record_id"] for r in first} & {r["record_id"] for r in second})
    assert first[0]["disease_name"] == "[演示]AMD"
    dates, diseases, daily = db.trend_counts()
    assert dates == {"2026-10-13": 205}
    assert diseases == {"Normal": 205}
    assert daily["2026-10-13"] == {"Normal": 205}
    db.delete_all()
    assert db.get_page(2) == ([], 0, 0)


def test_board_address_save_validation(tmp_path):
    import json
    path = tmp_path / "network.json"
    path.write_text(json.dumps({"network": {"camera_port": 5002}}))
    network_config.save_network_config("192.168.1.2", False, path=path, board_ip="192.168.1.3")
    saved = path.read_text()
    assert json.loads(saved)["network"]["board_ip"] == "192.168.1.3"
    assert json.loads(saved)["network"]["camera_port"] == 5002
    with pytest.raises(ValueError):
        network_config.save_network_config("192.168.1.2", False, path=path, board_ip="bad")
    assert path.read_text() == saved


def test_history_additive_migration_and_filtered_scope(tmp_path):
    import sqlite3
    path=tmp_path/'legacy.db'
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, record_id TEXT UNIQUE, timestamp TEXT, image_path TEXT, disease_name TEXT, confidence REAL)")
        conn.execute("INSERT INTO records VALUES (1,'legacy','2026-10-01 12:00:00','old.png','Normal',.8)")
    db=HistoryDB(path)
    assert Path(str(path)+'.before_batches.bak').is_file()
    db.add('new','2026-10-13 12:00:00','new.png','AMD',.9,{'AMD':.9},{'name':'fixture'},'batch1')
    rows,total,page,counts,daily=db.filtered({'start':'2026-10-13','end':'2026-10-14','batch':'batch1'})
    assert total==1 and counts=={'AMD':1} and daily=={'2026-10-13':1}
    assert json.loads(rows[0]['probabilities'])=={'AMD':.9}
    assert db.count()==2


def test_unified_export_structured_columns(tmp_path):
    from src.visualization.result_export import export_bundle
    import csv
    path=Path(export_bundle(tmp_path,'batch',[{'path':'sample.png','status':'success','save_status':'failed','disease_name':'Normal','confidence':.9,'error':'write failure'}],{},'fixture'))
    with (path/'results.csv').open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    assert rows[0]['save_status']=='failed' and rows[0]['disease_name']=='Normal'
    assert json.loads((path/'scope.json').read_text(encoding='utf-8'))['classification_success']==1
    assert (path/'report.pdf').stat().st_size>100


def test_windows_credential_roundtrip():
    import os
    if os.name!='nt':pytest.skip('Windows DPAPI')
    from src.core.credentials import encode_key,decode_key
    encoded=encode_key('test-fixture-not-a-real-key')
    assert encoded.startswith('dpapi:')
    assert decode_key(encoded)=='test-fixture-not-a-real-key'
