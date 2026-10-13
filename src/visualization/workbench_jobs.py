"""Window task controller: serialized model access without blocking Qt events."""
from datetime import datetime
from pathlib import Path
import json
import time
import uuid
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout
from .background_jobs import BackgroundJob
from .workbench_layout import update_probability_panel


class WorkbenchJobs:
    def _start_job(self, operation, callback, message, cancellable=False):
        if getattr(self, "_busy", False):
            return False
        self._busy = True
        self._closing = False
        self._job_callback = callback
        self._active_job = BackgroundJob(operation, self)
        self._active_job.completed.connect(self._job_completed)
        self._active_job.progress.connect(self._job_progress)
        self._active_job.finished.connect(self._job_finished)
        for button in (self.model_button,self.image_button,self.detect_button,self.batch_button):
            button.setEnabled(False)
        self.task_status_label.setText(message)
        self.status_bar.showMessage(message)
        self.task_progress.setRange(0,0)
        self.task_progress.show()
        self.cancel_task_button.setVisible(cancellable)
        self.cancel_task_button.setEnabled(True)
        self._active_job.start()
        return True

    def _job_completed(self, payload):
        if self._closing:
            return
        if payload["error"]:
            self.task_status_label.setText("任务失败")
            self.show_message_box("任务失败",payload["error"],QMessageBox.Warning)
        elif payload["cancelled"]:
            self.task_status_label.setText("任务已取消")
            self.status_bar.showMessage("任务已取消；已完成的有效记录保留")
        else:
            self._job_callback(payload["result"])

    def _job_finished(self):
        job = self._active_job
        self._active_job = None
        self._job_callback = None
        self._busy = False
        if hasattr(self,"batch_page") and self.batch_page.running:
            self.batch_page.running=False;self.batch_page.reload_batches();self.batch_page.render_batch()
        self.task_progress.hide()
        self.cancel_task_button.hide()
        ready = self.detector is not None and self.detector.model is not None
        self.model_button.setEnabled(True)
        self.image_button.setEnabled(True)
        self.batch_button.setEnabled(ready)
        self.detect_button.setEnabled(ready and self.current_image is not None)
        self.network_test_button.setEnabled(True)
        job.deleteLater()
        if self._closing:
            self.close()

    def _job_progress(self, payload):
        if self._closing: return
        done,total,name = payload[:3]
        if len(payload)>3:
            self._batch_messages.append(payload[3])
            if len(payload)>4:self._batch_counts=payload[4]
            if len(payload)>5:
                self.batch_page.pending[done-1]=payload[5]
                self.batch_page.render_batch()
            cursor=self.batch_output.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText(payload[3]+"\n")
            self.batch_output.setTextCursor(cursor)
        self.task_progress.setRange(0,total)
        self.task_progress.setValue(done)
        self.task_status_label.setText(f"{done}/{total}  ·  {name}")

    def cancel_current_task(self):
        if getattr(self, "_active_job", None) is not None:
            self._active_job.cancel()
            self.cancel_task_button.setEnabled(False)
            self.task_status_label.setText("正在结束当前步骤…")

    def closeEvent(self, event):
        if any(getattr(getattr(self,name,None),"jobs",[]) for name in ("batch_page","history_page")):
            self.status_bar.showMessage("正在完成查询或导出，请稍后关闭")
            event.ignore();return
        if getattr(self, "_busy", False):
            self._closing = True
            self._report_generation = getattr(self,"_report_generation",0)+1
            self.cancel_current_task()
            self.task_status_label.setText("正在结束任务，随后关闭窗口…")
            event.ignore()
            return
        if hasattr(self,"local_camera_page"):self.local_camera_page.stop()
        if hasattr(self,"_speech_player"):self._speech_player.stop()
        from .preferences import save_window
        save_window(self)
        super().closeEvent(event)

    def load_model(self, model_path=None):
        # Resolve application dependencies lazily to keep the controller import lightweight.
        import sys
        app = sys.modules[self.__class__.__module__]
        if getattr(self,"_busy",False): return
        if isinstance(model_path,bool): model_path=None
        if model_path is None:
            model_path,_=QFileDialog.getOpenFileName(self,"选择8类分类权重",str(app.APP_CONFIG.model_path.parent),"分类权重 (*.pt)")
            if not model_path: return
        if self.detector is None:
            self.detector=app.EyeDiseaseDetector(app.APP_CONFIG)
            self.result_processor=app.ResultProcessor(self.detector)
        self._clear_prediction_state()
        self.runtime_badge.setText("正在加载模型")
        self._start_job(lambda cancel,progress:self.detector.load_model(model_path),
                        self._model_loaded,"正在加载分类模型…")

    def _model_loaded(self, success):
        if not success:
            message=self.detector.last_error or "权重加载失败"
            self.model_info_label.setText("分类模型加载失败")
            self.runtime_badge.setText("模型不可用")
            self.task_status_label.setText("加载失败，请重新选择权重")
            self.show_message_box("模型加载失败",message,QMessageBox.Critical)
            return
        info=self.detector.model_info
        self.model_info_label.setText(f"{Path(info['weight_path']).name} · {info['device_name']}")
        self.model_info_label.setToolTip(json.dumps(info,ensure_ascii=False,indent=2))
        self.runtime_badge.setText("●  模型就绪  ·  " + ("CPU" if info['device']=='cpu' else "GPU"))
        self.task_status_label.setText("模型就绪，可以开始分类" if self.current_image is not None else "导入一张图像开始分析")
        self.status_bar.showMessage("分类模型已加载")
        self.batch_page.render_batch()

    def load_image(self):
        if getattr(self,"_busy",False): return
        from .image_directory import initial,remember
        path,_=QFileDialog.getOpenFileName(self,"选择图像",initial(self),"图像文件 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            remember(self,path)
            self.load_image_path(path)

    def load_image_path(self,path):
        if getattr(self,"_busy",False):return
        self._clear_prediction_state()
        self.current_image=None
        self.current_image_path=None
        self.original_image_label.clear()
        self.image_meta_label.setText("正在读取…")
        self.image_size_label.setText("— × — px")
        def read(cancel,progress):
            frame=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None: raise ValueError("无法读取这张图像，请检查文件是否损坏")
            h,w=frame.shape[:2]
            scale=min(1,1600/max(h,w))
            preview=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else frame
            return str(path),frame,preview
        self._start_job(read,self._image_loaded,"正在读取图像…")

    def _image_loaded(self, result):
        path,frame,preview=result
        self.current_image_path=path
        self.current_image=frame
        self.display_image(preview,self.original_image_label)
        self.image_meta_label.setText(Path(path).name)
        self.image_meta_label.setToolTip(path)
        self.image_size_label.setText(f"{frame.shape[1]} × {frame.shape[0]} px")
        self.task_status_label.setText("图像已就绪")
        self.status_bar.showMessage("图像已加载")

    def detect_image(self):
        import sys
        app = sys.modules[self.__class__.__module__]
        if getattr(self,"_busy",False): return
        self._clear_prediction_state()
        if self.current_image is None or self.detector is None or self.detector.model is None:
            self._classification_failed("请先加载模型和有效图像")
            return
        generation=self._report_generation
        frame=self.current_image
        config=app.APP_CONFIG
        db=app.get_history_db()
        def classify(cancel,progress):
            started=time.perf_counter()
            outcome=self.detector.classify(frame)
            seconds=time.perf_counter()-started
            warning=None
            if outcome.success and not cancel.is_set():
                try:
                    config.image_dir.mkdir(parents=True,exist_ok=True)
                    path=config.image_dir/("temp_image_"+uuid.uuid4().hex+".png")
                    ok,encoded=cv2.imencode('.png',frame)
                    if not ok:raise ValueError("图像编码失败")
                    encoded.tofile(str(path))
                    db.add(str(uuid.uuid4()),datetime.now().strftime('%Y-%m-%d %H:%M:%S'),str(path),outcome.disease_name,outcome.confidence,outcome.probabilities,self.detector.model_info)
                except Exception as error:
                    warning=f"分类成功，历史记录保存失败：{error}"
            return generation,outcome,seconds,warning
        self._start_job(classify,self._classification_ready,"正在分析图像…",True)

    def _classification_ready(self,result):
        generation,outcome,seconds,warning=result
        if generation!=self._report_generation:return
        if not outcome.success:
            self._classification_failed(outcome.error)
            return
        self.current_results=outcome.raw_results
        self.current_disease=outcome.disease_name
        self.current_confidence=outcome.confidence
        self.all_classes_confidence=outcome.probabilities
        self.result_processor.current_disease=outcome.disease_name
        self.result_processor.current_confidence=outcome.confidence
        self._update_classification_summary()
        update_probability_panel(self,outcome.probabilities)
        self.inference_time_label.setText(f"本次推理 {seconds:.2f}s")
        self.results_button.setEnabled(True)
        self.advice_button.setEnabled(True)
        self.task_status_label.setText("分析完成" if not warning else "分析完成，历史保存失败")
        from .rich_text import advice_guide
        self.advice_text.setHtml(advice_guide(classified=True))
        self.status_bar.showMessage(warning or "分类完成；可查看全部概率或生成辅助说明")

    def test_network_and_show_result(self):
        import sys
        app = sys.modules[self.__class__.__module__]
        if getattr(self, "_busy", False): return
        key = self.api_key_input.text().strip()
        if not key and self.deepseek_api is not None: key = self.deepseek_api.api_key
        if not key:
            self.show_message_box("连接测试", "请先填写或保存 API Key", QMessageBox.Information)
            return
        client = app.DeepSeekAPI(key, config=app.APP_CONFIG)
        self.network_test_button.setEnabled(False)
        def finished(result):
            self.network_test_button.setEnabled(True)
            self._network_checked(result)
        self.batch_page.submit(lambda cancel,progress:client.request_custom_advice("请简单回复连接测试成功"),finished)

    def _network_checked(self, result):
        self.task_status_label.setText("API 连接成功" if result.success else "API 连接失败")
        self.show_message_box("连接测试", "DeepSeek 已成功返回回答" if result.success else result.content,
                              QMessageBox.Information if result.success else QMessageBox.Warning)

    def batch_process(self):
        import sys,shutil
        app=sys.modules[self.__class__.__module__]
        if getattr(self,"_busy",False):return
        if self.detector is None or self.detector.model is None:
            self.show_message_box("模型未就绪","请先加载模型");return
        page=self.batch_page
        paths=[r['path'] for r in page.pending if r['status']=='pending']
        if not paths:
            self.show_message_box("没有待处理图像","请先添加图像，或选择重试失败项。");return
        bid=uuid.uuid4().hex;db=app.get_history_db();info=dict(self.detector.model_info)
        try:db.create_batch(bid,paths,info)
        except Exception as error:
            self.show_message_box("无法创建批次",str(error));return
        page.batch_id=bid;page.pending=[{'path':p,'status':'pending','save_status':'pending'} for p in paths];page.running=True
        self._batch_messages=[];self._batch_counts={};self.workspace_pages.setCurrentIndex(1);page.render_batch()
        def batch(cancel,progress):
            counts,messages={},[]
            try:
                for index,path in enumerate(paths):
                    if cancel.is_set():break
                    started=time.perf_counter()
                    row={'path':path,'batch_id':bid,'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'status':'failed','save_status':'pending','model_info':info}
                    try:
                        _,outcome=next(self.detector.classify_files([path]))
                        row.update(seconds=time.perf_counter()-started,probabilities=outcome.probabilities)
                        if outcome.success:
                            row.update(status='success',disease_name=outcome.disease_name,confidence=outcome.confidence)
                            counts[outcome.disease_name]=counts.get(outcome.disease_name,0)+1
                            try:
                                app.APP_CONFIG.image_dir.mkdir(parents=True,exist_ok=True)
                                archived=app.APP_CONFIG.image_dir/(uuid.uuid4().hex+Path(path).suffix)
                                shutil.copy2(path,archived)
                                db.add(uuid.uuid4().hex,row['timestamp'],str(archived),outcome.disease_name,outcome.confidence,outcome.probabilities,info,bid)
                                row['save_status']='saved';row['image_path']=str(archived)
                            except Exception as error:row.update(save_status='failed',error=str(error))
                        else:row['error']=outcome.error
                    except Exception as error:row['error']=str(error)
                    db.save_batch_item(bid,index,row)
                    message=f"{Path(path).name}: {row['status']} / {row['save_status']}"
                    messages.append(message)
                    progress((index+1,len(paths),Path(path).name,message,dict(counts),row))
                db.finish_batch(bid,'cancelled' if cancel.is_set() else 'completed')
            except Exception:
                db.finish_batch(bid,'interrupted');raise
            return counts,messages
        self._start_job(batch,self._batch_ready,"正在批量分类…",True)

    def _batch_ready(self,result):
        self.task_status_label.setText("批量分类完成")
        self.status_bar.clearMessage()
        self._last_batch_report=result
        self.batch_output.setPlainText("\n".join(result[1]))
        self._batch_messages=list(result[1])

    def show_hardware_workspace(self):
        self.image_tab_widget.setCurrentIndex(1)
