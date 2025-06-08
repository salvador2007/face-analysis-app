import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

PROJECT_DIR = os.path.abspath(".")  # 현재 폴더
IGNORE_EXT = ['.tmp', '.swp', '.DS_Store']  # 무시할 확장자

class GitHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if any(event.src_path.endswith(ext) for ext in IGNORE_EXT):
            return

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 변경 감지됨: {event.src_path}")
        try:
            os.system("git add .")
            commit_msg = f"Auto Commit at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            os.system(f'git commit -m "{commit_msg}"')
            os.system("git push")
            print("[INFO] 자동 커밋 및 푸시 완료!\n")
        except Exception as e:
            print(f"[ERROR] Git 작업 실패: {e}")

if __name__ == "__main__":
    print("[INFO] Git 자동 감시 시작...")
    event_handler = GitHandler()
    observer = Observer()
    observer.schedule(event_handler, PROJECT_DIR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

