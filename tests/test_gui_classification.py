from dataclasses import replace
from unittest.mock import Mock
import numpy as np
import pytest
import time
from PyQt5.QtTest import QTest
from PyQt5.QtCore import QTimer
import src.visualization.workbench_jobs as jobs

import visualization_test2 as gui
from src.core.classification import EyeDiseaseDetector
from src.core.history import HistoryDB
from test_application_services import FakeModel


def wait_for_job(w):
    deadline = time.monotonic() + 5
    while getattr(w, "_busy", False) and time.monotonic() < deadline:
        gui.QApplication.processEvents()
        QTest.qWait(5)
    assert not getattr(w, "_busy", False), "background task did not finish"
    wait_for_pages(w)


@pytest.fixture(scope="module")
def app():
    return gui.QApplication.instance() or gui.QApplication([])


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.QTimer, "singleShot", Mock())
    config = replace(gui.APP_CONFIG, image_dir=tmp_path / "images", history_db=tmp_path / "history.db", enable_voice=False, enable_board=False)
    monkeypatch.setattr(gui, "APP_CONFIG", config)
    history = HistoryDB(config.history_db)
    monkeypatch.setattr(gui, "get_history_db", lambda: history)
    original_build=gui.build_workbench
    def isolated_build(w,config):
        original_build(w,config)
        w._ui_preferences_path=tmp_path/"ui_preferences.json"
    monkeypatch.setattr(gui,"build_workbench",isolated_build)
    w = gui.MainWindow()
    w._ui_preferences_path=tmp_path/"ui_preferences.json"
    w.show_message_box = Mock()
    w.show_disease_result = Mock()
    w.display_results = Mock()
    yield w, history
    wait_for_job(w)
    wait_for_pages(w)
    w.close()


def install_detector(window, tmp_path, model):
    w, _ = window
    path = tmp_path / "fake.pt"
    path.write_bytes(b"test fixture")
    w.detector = EyeDiseaseDetector(model_factory=lambda _: model)
    assert w.detector.load_model(path)
    w.result_processor = gui.ResultProcessor(w.detector)
    w.current_image = np.zeros((12, 12, 3), dtype=np.uint8)


def test_failed_classification_clears_previous_success_without_history_or_advice(window, tmp_path):
    w, history = window
    install_detector(window, tmp_path, FakeModel([0] * 8))
    w.current_disease, w.current_confidence = "AMD", 0.98
    w.detected_image_label.setText("previous prediction")
    w.results_button.setEnabled(True)
    w.advice_button.setEnabled(True)
    gui.QTimer.singleShot.reset_mock()
    w.detect_image()
    wait_for_job(w)
    assert history.count() == 0 and w.current_disease is None
    assert not w.results_button.isEnabled() and not w.advice_button.isEnabled()
    assert w.detected_image_label.text() == "等待有效分类结果"
    gui.QTimer.singleShot.assert_not_called()
    w.show_disease_result.assert_not_called()


def test_valid_classification_displays_saves_and_enables_advice(window, tmp_path):
    w, history = window
    install_detector(window, tmp_path, FakeModel())
    w.detect_image()
    wait_for_job(w)
    assert w.current_disease == "Normal" and history.count() == 1
    assert w.results_button.isEnabled() and w.advice_button.isEnabled()
    w.show_disease_result.assert_not_called()
    assert w.classification_summary_label.text() == "正常"
    assert w.probability_rows[0][0].text() == "正常"
    w.show_results()
    w.show_disease_result.assert_not_called()
    assert len(w.probability_rows) == 8
    assert list(gui.APP_CONFIG.image_dir.glob("*.png"))


def test_failed_model_switch_disables_image_and_batch_actions(window, tmp_path):
    w, _ = window
    install_detector(window, tmp_path, FakeModel())
    w.load_model(str(tmp_path / "missing.pt"))
    wait_for_job(w)
    assert w.detector.model is None
    assert not w.detect_button.isEnabled() and not w.batch_button.isEnabled()


def test_settings_panel_has_network_controls(window):
    w, _ = window
    panel = w.create_runtime_settings_dialog()
    assert panel.findChild(gui.QLineEdit, "network_ip") is not None
    assert panel.findChild(gui.QCheckBox, "network_auto") is not None
    assert panel.findChild(gui.QComboBox, "network_interfaces") is not None
    panel.close()


def test_batch_invalid_results_are_not_saved(window, tmp_path, monkeypatch):
    w, history = window
    model = FakeModel([0] * 8)
    model.predict = Mock(return_value=model.results)
    install_detector(window, tmp_path, model)
    class FileDialog:
        ExistingFiles = 1
        Accepted = 1
        def setFileMode(self, *_): pass
        def setNameFilter(self, *_): pass
        def exec(self): return self.Accepted
        def selectedFiles(self): return [str(tmp_path / "invalid.png")]
    monkeypatch.setattr(jobs, "QFileDialog", FileDialog)
    w.show_batch_report = Mock()
    w.batch_page.add_paths([str(tmp_path / "invalid.png")])
    w.batch_process()
    wait_for_job(w)
    assert history.count() == 0
    counts, messages = w._last_batch_report
    w.show_batch_report.assert_not_called()
    assert messages[0] in w.batch_output.toPlainText()
    assert not counts and w.batch_page.pending[0]["status"] == "failed"


def test_gui_api_disconnect_posts_error_instead_of_success(window, monkeypatch):
    import requests
    from src.core.llm import DeepSeekAPI
    w, history = window
    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline test")
    w.deepseek_api = DeepSeekAPI("test-key", post=offline, sleep=lambda _: None)
    w._last_user_question = "测试问题"
    events = []
    monkeypatch.setattr(gui.QApplication, "postEvent", lambda _, event: events.append(event))
    w.process_ai_response()
    assert events[-1].event_type == "error"
    assert not any(event.event_type == "completed" for event in events)
    assert history.count() == 0


def test_late_advice_reply_is_ignored_after_classification_failure(window):
    from src.core.llm import AdviceResult
    w, _ = window
    w.current_results = []
    w.current_disease, w.current_confidence = "Normal", 0.8
    w._report_generation = 1
    payload = {"generation": 1, "signature": ("Normal", 0.8, id(w.current_results)),
               "advice": AdviceResult(True, "old response", "deepseek")}
    w._clear_prediction_state()
    assert not w._apply_report_reply(payload)
    assert "old response" not in w.advice_text.toPlainText()


def test_inference_is_off_ui_thread_and_cancelled_output_is_not_saved(window, tmp_path):
    import threading
    w, history = window
    model = FakeModel()
    entered, release = threading.Event(), threading.Event()
    worker_ids = []
    def slow_predict(*args, **kwargs):
        worker_ids.append(threading.get_ident())
        entered.set()
        release.wait(5)
        return model.results
    model.predict = slow_predict
    install_detector(window, tmp_path, model)
    main_id = threading.get_ident()
    w.detect_image()
    try:
        assert entered.wait(1)
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(5)
        deadline = time.monotonic() + 1
        while not ticks and time.monotonic() < deadline:
            QTest.qWait(10)
        timer.stop()
        assert ticks and worker_ids[0] != main_id
        assert w._busy and not w.model_button.isEnabled()
        w.cancel_current_task()
    finally:
        release.set()
        wait_for_job(w)
    assert history.count() == 0 and w.current_disease is None
    assert not w.results_button.isEnabled()


def test_corrupt_image_clears_previous_result_and_prevents_classification(window, tmp_path):
    w, history = window
    install_detector(window, tmp_path, FakeModel())
    corrupt = tmp_path / "坏图.png"
    corrupt.write_bytes(b"not an image")
    w.load_image_path(str(corrupt))
    wait_for_job(w)
    assert w.current_image is None and w.current_disease is None
    assert not w.detect_button.isEnabled() and history.count() == 0


def test_late_classification_does_not_replace_invalidated_image(window, tmp_path):
    w, _ = window
    install_detector(window, tmp_path, FakeModel())
    result = w.detector.classify(w.current_image)
    w._report_generation = 10
    w._clear_prediction_state()
    w._classification_ready((10, result, .01, None))
    assert w.current_disease is None and not w.results_button.isEnabled()


def test_window_close_waits_for_worker_and_discards_cancelled_result(window, tmp_path):
    import threading
    w, history = window
    model = FakeModel()
    entered, release = threading.Event(), threading.Event()
    def slow_predict(*args, **kwargs):
        entered.set()
        release.wait(5)
        return model.results
    model.predict = slow_predict
    install_detector(window, tmp_path, model)
    w.detect_image()
    try:
        assert entered.wait(1)
        assert not w.close()
        assert w._closing and w._active_job.cancelled.is_set()
    finally:
        release.set()
        wait_for_job(w)
    assert history.count() == 0 and w.current_disease is None


def test_connection_failure_does_not_become_success_due_to_long_error_text(window, monkeypatch):
    from src.core.llm import AdviceResult
    w, _ = window
    w.api_key_input.setText("test-key")
    client = Mock()
    client.request_custom_advice.return_value = AdviceResult(False, "连接失败：此处是一段超过十个字符的错误信息", "error", "offline")
    monkeypatch.setattr(gui, "DeepSeekAPI", lambda *args, **kwargs: client)
    w.test_network_and_show_result()
    wait_for_job(w)
    assert w.task_status_label.text() == "API 连接失败"
    assert w.show_message_box.call_args[0][2] == gui.QMessageBox.Warning


def test_font_size_changes_live_and_persists_without_changing_model_config(window, tmp_path):
    import json
    w, _ = window
    w._ui_preferences_path = tmp_path / "ui_preferences.json"
    w.font_size_combo.setCurrentIndex(w.font_size_combo.findData(14))
    w.font_size_combo.setCurrentIndex(w.font_size_combo.findData(18))
    assert "font-size: 18pt" in w.styleSheet()
    assert json.loads(w._ui_preferences_path.read_text(encoding="utf-8"))["font_size"] == 18
    assert w.right_tab_widget.count() == 2 and w.image_tab_widget.count() == 3
    assert not w.board_interaction_button.isHidden() and not w.voice_input_button.isHidden()


def test_connection_settings_embedded_once(window):
    w, _ = window
    w.ensure_connection_settings(2)
    panel = w.connection_settings
    assert panel.findChild(gui.QLineEdit, "network_ip") is not None
    assert panel.findChild(gui.QLineEdit, "board_ip") is not None
    w.right_tab_widget.setCurrentIndex(0)
    w.ensure_connection_settings(2)
    assert w.connection_settings is panel
    auto = panel.findChild(gui.QCheckBox, "network_auto")
    auto.setChecked(False)
    assert panel.findChild(gui.QLineEdit, "network_ip").isEnabled()
    auto.setChecked(True)
    assert not panel.findChild(gui.QLineEdit, "network_ip").isEnabled()


def test_microphone_disabled_reports_without_starting_job(window):
    w, _ = window
    w.test_microphone()
    w.show_message_box.assert_called_once()
    assert not getattr(w, "_busy", False)


def test_workspace_pages_reuse_history_and_hide_duplicate_results(window):
    w, history = window
    assert [w.workspace_pages.tabText(i) for i in range(3)] == ["单图分析", "批量分析", "历史记录"]
    assert w.results_button.isHidden()
    history.add("page-record", "2026-10-13 12:00:00", "image.png", "Normal", .8)
    w.workspace_pages.setCurrentIndex(2)
    wait_for_pages(w)
    table = w.history_page.table
    assert table.rowCount() == 1
    w.workspace_pages.setCurrentIndex(0)
    w.workspace_pages.setCurrentIndex(2)
    wait_for_pages(w)
    assert w.history_page.table is table
    assert w.right_tab_widget.count() == 2


def test_chat_collapses_probabilities_and_keeps_summary(window):
    w, _ = window
    assert not w.probability_toggle.isChecked()
    w.probability_toggle.setChecked(True)
    assert w.probability_toggle.isChecked()
    w.right_tab_widget.setCurrentIndex(1)
    assert w.probability_details.isHidden()
    assert not w.classification_summary_label.isHidden()
    w.probability_toggle.setChecked(True)
    assert not w.probability_details.isHidden()
    assert w.chat_input.maximumHeight() == 58


def test_batch_and_history_chart_exports(window, monkeypatch, tmp_path):
    import csv
    w, db = window
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    def export_from_dialog(dialog):
        buttons = [b for b in dialog.findChildren(gui.QPushButton) if b.text() == "导出统计图与数据"]
        assert len(buttons) == 1
        buttons[0].click()
        return 0
    monkeypatch.setattr(gui.QDialog, "exec_", export_from_dialog)
    w.show_batch_report({"Normal": 2}, ["Normal: test fixture"])
    batch = next(tmp_path.glob("batch_*"))
    assert (batch / "chart_1.png").stat().st_size > 100
    assert (batch / "chart_1.pdf").stat().st_size > 100
    with (batch / "statistics.csv").open(encoding="utf-8-sig") as stream:
        assert list(csv.reader(stream))[1] == ["Normal", "2"]
    db.add("real", "2026-10-13 12:00:00", "fixture.png", "Normal", .8)
    db.add("demo", "2026-10-13 12:00:00", "fixture.png", "[演示]AMD", .8)
    w.show_trend_analysis()
    history = next(tmp_path.glob("history_*"))
    assert len(list(history.glob("chart_*.png"))) == 3
    with (history / "statistics.csv").open(encoding="utf-8-sig") as stream:
        assert list(csv.reader(stream))[1:] == [["2026-10-13", "Normal", "1"]]


def wait_for_pages(w):
    deadline=time.monotonic()+10
    while (w.batch_page.jobs or w.history_page.jobs) and time.monotonic()<deadline:
        gui.QApplication.processEvents();QTest.qWait(5)
    assert not w.batch_page.jobs and not w.history_page.jobs


def test_chat_old_reply_is_ignored_after_reset(window):
    from src.visualization.conversation import reset,finish
    from src.core.llm import AdviceResult
    w,_=window
    reset(w);generation=w._chat_generation
    w.advice_text.setPlainText("preserve advice")
    reset(w)
    finish(w,(generation,"old question",AdviceResult(True,"old answer","api")))
    assert "old answer" not in w.chat_display.toPlainText()
    assert w.advice_text.toPlainText()=="preserve advice"
    assert w._chat_client is None


def test_batch_success_count_independent_of_history_save(window,tmp_path,monkeypatch):
    import cv2
    w,db=window
    install_detector(window,tmp_path,FakeModel())
    image=tmp_path/'sample.png';cv2.imwrite(str(image),np.zeros((12,12,3),dtype=np.uint8))
    monkeypatch.setattr(db,'add',Mock(side_effect=OSError('database write test')))
    w.batch_page.add_paths([str(image)]);w.batch_process();wait_for_job(w)
    row=w.batch_page.pending[0]
    assert row['status']=='success' and row['save_status']=='failed'
    assert w._batch_counts=={'Normal':1}
    saved=db.batch_items(w.batch_page.batch_id)
    assert saved[0]['save_status']=='failed'
    assert db.list_batches()[0]['state']=='completed'


def test_font_preference_saved_and_applied(window,tmp_path):
    import json
    from src.visualization.workbench_layout import change_font_size
    w,_=window;w._ui_preferences_path=tmp_path/'fonts.json'
    change_font_size(w)
    family=w.font_family_combo.currentData()
    assert family in w.styleSheet()
    assert json.loads(w._ui_preferences_path.read_text(encoding='utf-8'))['font_family']==family
    assert w.chat_display.document().defaultFont().family()==family

def test_eight_class_chart_and_table_follow_large_font(window,tmp_path):
    from src.visualization.workbench_layout import NAMES
    from src.visualization.workbench_theme import apply_workbench_theme
    w,_=window
    w._ui_preferences_path=tmp_path/"chart_fonts.json"
    w.font_size_combo.setCurrentIndex(w.font_size_combo.findData(18))
    apply_workbench_theme(w)
    for page in (w.batch_page,w.history_page):
        page.chart.set_counts(dict.fromkeys(NAMES,1))
        assert page.chart.minimumHeight() >= 70+8*(page.chart.fontMetrics().height()+30)
        assert page.table.verticalHeader().defaultSectionSize() >= page.table.fontMetrics().height()+18
        assert page.chart_choice.count() >= 2
        page.chart_choice.setCurrentIndex(1)
        assert page.chart.mode==1
    assert w.history_page.chart_choice.count()==3
    assert w.history_page.start.width() >= w.history_page.start.fontMetrics().horizontalAdvance('YYYY-MM-DD')+32




def test_desktop_preferences_restore_and_reading_sizes(window):
    from src.visualization.preferences import restore,save_window
    from src.visualization.image_directory import read,save
    from src.visualization.rich_text import render_text
    w,_=window
    w.reading_size_combo.setCurrentIndex(w.reading_size_combo.findData(20))
    w.voice_chat_enabled.setChecked(True)
    save_window(w)
    settings=read(w)
    assert settings['reading_size']==20 and settings['auto_read'] is True
    assert settings['window_geometry'] and settings['splitter_state']
    save(w,reading_size=18,auto_read=False)
    restore(w)
    assert w.reading_size_combo.currentData()==18
    assert not w.voice_chat_enabled.isChecked()
    w.advice_text.setHtml(render_text('# Heading\nBody'))
    body=w.advice_text.document().lastBlock().begin().fragment()
    assert body.charFormat().fontPointSize()==18
