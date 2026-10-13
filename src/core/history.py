"""SQLite history storage, independent of Qt and model loading."""
import os,json,sqlite3,uuid
from datetime import datetime, timedelta
from pathlib import Path
from configs.runtime_config import load_app_config
from .classification import valid_prediction

class HistoryDB:
    """轻量级 SQLite 历史记录管理器, 替代 JSON 文件存储"""

    def __init__(self, db_path=None):
        self.db_path = str(Path(db_path or load_app_config().history_db).expanduser().resolve())
        self.db_dir = str(Path(self.db_path).parent)
        os.makedirs(self.db_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        # One-time SQLite-consistent backup before the additive schema upgrade.
        backup=Path(self.db_path+".before_batches.bak")
        if Path(self.db_path).is_file() and not backup.exists():
            with sqlite3.connect(self.db_path) as source:
                columns=[r[1] for r in source.execute("PRAGMA table_info(records)")]
                if columns and "batch_id" not in columns:
                    with sqlite3.connect(str(backup)) as destination:source.backup(destination)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    disease_name TEXT NOT NULL,
                    confidence REAL NOT NULL
                )
            """)

            # 向后兼容：为旧数据库添加 advice 字段
            try:
                conn.execute("ALTER TABLE records ADD COLUMN advice TEXT")
            except sqlite3.OperationalError:
                pass  # 字段已存在，忽略

            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON records(timestamp DESC)")
            for name in ("probabilities", "model_info", "batch_id"):
                if name not in [r[1] for r in conn.execute("PRAGMA table_info(records)")]:
                    conn.execute(f"ALTER TABLE records ADD COLUMN {name} TEXT")
            conn.execute("CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, created TEXT, state TEXT, model_info TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS batch_items (batch_id TEXT, position INTEGER, payload TEXT, PRIMARY KEY(batch_id,position))")
            conn.commit()

    def add(self, record_id, timestamp, image_path, disease_name, confidence, probabilities=None, model_info=None, batch_id=None):
        # Explicitly labelled demonstration records remain separate from predictions.
        is_demo = isinstance(disease_name, str) and disease_name.startswith("[演示]")
        checked_name = disease_name[len("[演示]"):].strip() if is_demo else disease_name
        if not valid_prediction(checked_name, confidence):
            raise ValueError("只有有效分类结果或明确标记的演示记录才能入库")
        confidence = float(confidence)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO records (record_id, timestamp, image_path, disease_name, confidence) VALUES (?,?,?,?,?)",
                (record_id, timestamp, image_path, disease_name, round(confidence, 4))
            )
            conn.execute("UPDATE records SET probabilities=?, model_info=?, batch_id=? WHERE record_id=?",
                         (json.dumps(probabilities, ensure_ascii=False) if probabilities is not None else None,
                          json.dumps(model_info, ensure_ascii=False) if model_info is not None else None, batch_id, record_id))
            conn.commit()

    def create_batch(self, batch_id, paths, model_info):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO batches VALUES (?,?,?,?)",(batch_id,datetime.now().isoformat(timespec="seconds"),"running",json.dumps(model_info,ensure_ascii=False)))
            conn.executemany("INSERT INTO batch_items VALUES (?,?,?)",[(batch_id,i,json.dumps({"path":p,"status":"pending","save_status":"pending"},ensure_ascii=False)) for i,p in enumerate(paths)])

    def save_batch_item(self, batch_id, position, item):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE batch_items SET payload=? WHERE batch_id=? AND position=?",(json.dumps(item,ensure_ascii=False),batch_id,position))

    def finish_batch(self, batch_id, state):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE batches SET state=? WHERE batch_id=?",(state,batch_id))

    def list_batches(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory=sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM batches ORDER BY created DESC LIMIT 200")]

    def batch_stats(self):
        """批次列表：登记张数、实际入库张数、状态与创建时间。

        失败项不写 records，因此 planned 与 saved 的差值即为失败或跳过的张数。
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT b.batch_id AS batch_id, b.created AS created, b.state AS state,"
                " (SELECT COUNT(*) FROM batch_items i WHERE i.batch_id = b.batch_id) AS planned,"
                " (SELECT COUNT(*) FROM records r WHERE r.batch_id = b.batch_id) AS saved"
                " FROM batches b ORDER BY b.created DESC, b.batch_id DESC LIMIT 200"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_batch(self, batch_id):
        """删除批次登记与批次项；已入库的记录保留，避免误删历史。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM batch_items WHERE batch_id = ?", (batch_id,))
            conn.execute("DELETE FROM batches WHERE batch_id = ?", (batch_id,))
            conn.commit()

    def batch_items(self, batch_id):
        with sqlite3.connect(self.db_path) as conn:
            return [json.loads(r[0]) for r in conn.execute("SELECT payload FROM batch_items WHERE batch_id=? ORDER BY position",(batch_id,))]

    def filtered(self, filters, page=0, page_size=100, export=False):
        clauses=["disease_name NOT LIKE '[对话]%'"];args=[]
        for key,column in (("start","timestamp >="),("end","timestamp <"),("disease","disease_name ="),("batch","batch_id =")):
            if filters.get(key):clauses.append(column+" ?");args.append(filters[key])
        if filters.get("real_only",True):clauses.append("disease_name NOT LIKE '[演示]%'")
        if filters.get("search"):
            clauses.append("instr(lower(image_path),lower(?)) > 0");args.append(filters['search'])
        where=" AND ".join(clauses)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory=sqlite3.Row;conn.execute("BEGIN")
            total=conn.execute("SELECT COUNT(*) FROM records WHERE "+where,args).fetchone()[0]
            page=max(0,min(page,max(0,(total-1)//page_size)))
            query="SELECT * FROM records WHERE "+where+" ORDER BY timestamp DESC,id DESC"
            rows=conn.execute(query if export else query+" LIMIT ? OFFSET ?",args if export else args+[page_size,page*page_size]).fetchall()
            counts=dict(conn.execute("SELECT disease_name,COUNT(*) FROM records WHERE "+where+" GROUP BY disease_name",args).fetchall())
            daily=dict(conn.execute("SELECT substr(timestamp,1,10),COUNT(*) FROM records WHERE "+where+" GROUP BY substr(timestamp,1,10) ORDER BY substr(timestamp,1,10)",args).fetchall())
        return [dict(r) for r in rows],total,page,counts,daily

    def get_all(self, limit=10000):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM records WHERE disease_name NOT LIKE '[对话]%' ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_page(self, page=0, page_size=100):
        """Bounded, newest-first page; count and rows use one read snapshot."""
        page_size = max(1, min(int(page_size), 200))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            total = conn.execute("SELECT COUNT(*) FROM records WHERE disease_name NOT LIKE '[对话]%'").fetchone()[0]
            page = max(0, min(int(page), max(0, (total - 1) // page_size)))
            rows = conn.execute("SELECT * FROM records WHERE disease_name NOT LIKE '[对话]%' ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?", (page_size, page * page_size)).fetchall()
        return [dict(row) for row in rows], total, page

    def trend_counts(self):
        """Aggregate real classifications without loading image paths or advice."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT substr(timestamp,1,10), disease_name, COUNT(*) FROM records WHERE disease_name NOT LIKE '[演示]%' AND disease_name NOT LIKE '[对话]%' GROUP BY substr(timestamp,1,10), disease_name").fetchall()
        dates, diseases, daily = {}, {}, {}
        for day, disease, count in rows:
            try:
                datetime.strptime(day, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            if not valid_prediction(disease, 0.5):
                continue
            dates[day] = dates.get(day, 0) + count
            diseases[disease] = diseases.get(disease, 0) + count
            daily.setdefault(day, {})[disease] = count
        return dates, diseases, daily

    def delete_by_record_id(self, record_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
            conn.commit()

    def delete_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM records")
            conn.commit()

    def update_advice(self, record_id, advice):
        """将 AI 建议存入数据库，实现本地缓存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE records SET advice = ? WHERE record_id = ?", (advice, record_id))
            conn.commit()

    def count(self):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def migrate_from_json(self, json_path):
        """从旧版 JSON 文件迁移数据到 SQLite"""
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for r in data:
                    self.add(
                        r.get("record_id", str(uuid.uuid4())),
                        r.get("timestamp", ""),
                        r.get("image_path", ""),
                        r.get("disease_name", ""),
                        r.get("confidence", 0)
                    )
                # 迁移后重命名旧文件
                os.rename(json_path, json_path + ".bak")
                print(f"[HistoryDB] 已从 JSON 迁移 {len(data)} 条记录到 SQLite")
            except Exception as e:
                print(f"[HistoryDB] JSON 迁移失败: {e}")

    def generate_demo_data(self):
        """生成 30 天模拟检测数据，用于趋势分析演示"""
        import random
        db = self

        diseases = ["Diabetic Retinopathy", "Normal", "Myopia",
                     "AMD", "Cataract", "Glaucoma", "Hypertensive Retinopathy", "Other"]
        # 演示抽样权重，不代表真实疾病分布
        weights = [50, 20, 15, 5, 4, 3, 3, 2]

        base_date = datetime.now() - timedelta(days=30)
        # 清除旧的演示数据（直接用 SQL，避开 get_all 的 LIMIT 500 限制）
        with sqlite3.connect(db.db_path) as conn:
            conn.execute("DELETE FROM records WHERE disease_name LIKE '[演示]%'")
            conn.commit()

        records_added = 0
        for day_offset in range(30):
            date = base_date + timedelta(days=day_offset)
            # 每天 15-30 条记录，周末少一些
            day_count = random.randint(150, 250) if date.weekday() < 5 else random.randint(50, 100)
            for _ in range(day_count):
                disease = random.choices(diseases, weights=weights, k=1)[0]
                confidence = round(random.uniform(0.82, 0.99), 4)
                timestamp = date.strftime("%Y-%m-%d") + f" {random.randint(8,18):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
                db.add(
                    record_id=str(uuid.uuid4()),
                    timestamp=timestamp,
                    image_path=f"demo/eye_scan_{day_offset}_{records_added}.jpg",
                    disease_name=f"[演示]{disease}",
                    confidence=confidence
                )
                records_added += 1

        return records_added


_history_db = None

def get_history_db():
    global _history_db
    if _history_db is None:
        _history_db = HistoryDB()
    return _history_db
