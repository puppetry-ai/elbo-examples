from elbo.tracker.tracker import TaskTracker

if __name__ == "__main__":
    tracker = TaskTracker("Hello World")
    tracker.log_message("Hi there! 👋")
    tracker.upload_logs()
