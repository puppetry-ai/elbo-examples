from elbo.tracker.tracker import TaskTracker

if __name__ == "__main__":
    # Create the tracker
    tracker = TaskTracker("Hello World")

    # Log a message
    tracker.log_message("Hi there! 👋")

    # Log a metric
    tracker.log_key_metric("Accuracy", 100.0)

    # Log some images
    tracker.log_image("An AI generated image of a Cat 🐱", "images/aicat.png")

    # Upload logs, thats it!
    tracker.upload_logs()
