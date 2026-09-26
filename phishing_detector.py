import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# 1. DATASET
# ============================================================

def get_dataset():
    """
    Returns a small sample dataset containing
    Phishing and Safe email examples.
    """

    phishing_emails = [
        "URGENT: Your account has been suspended! Click here http://login-verify-bank.com to restore access.",
        "Dear customer, update your billing information immediately at http://paypal-verify-secure.net or lose your account.",
        "Congratulations! You won $10,000. Claim your prize now at http://bit.ly/claim-prize-free",
        "Security Alert: Unusual login detected. Verify your password now: http://security-update-account.com",
        "Action Required: Your Office 365 password expires today. Change it here: http://microsoft-update-pass.org",
        "Your package delivery failed. Confirm your shipping address: http://fedex-tracking-parcel.info",
        "Bank Notice: Unauthorized transaction of $500 detected. Cancel transaction here http://bank-secure-alert.xyz",
        "Urgent requirement: Wire transfer needed for urgent business request. Click http://wire-transfer-login.com",
        "Your Apple ID is locked due to multiple failed attempts. Unlock at http://apple-id-verify.co",
        "Tax Refund Notification: You are eligible for a refund of $850. Submit form at http://irs-refund-portal.net"
    ]

    safe_emails = [
        "Hi Team, please find attached the meeting notes from yesterday's discussion.",
        "Reminder: The weekly project review meeting is scheduled for tomorrow at 10 AM on Teams.",
        "Thanks for your order! Your receipt and order confirmation details are attached.",
        "Hey, are we still meeting for lunch today at 1 PM?",
        "Please review the updated project schedule and let me know if you have any feedback.",
        "Here is the monthly performance report for August. Great work everyone!",
        "Can you send over the final draft of the presentation before 5 PM?",
        "Your subscription renewal invoice is available in your account dashboard.",
        "Hi John, happy birthday! Hope you have a fantastic day with family.",
        "The server maintenance has been completed successfully with no downtime."
    ]

    X = phishing_emails + safe_emails

    y = (
        ["Phishing"] * len(phishing_emails)
        + ["Safe"] * len(safe_emails)
    )

    return X, y


# ============================================================
# 2. GUI APPLICATION
# ============================================================

class PhishingDetectorApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Thiranax Cybersecurity - Phishing Email Detector"
        )

        self.root.geometry("820x680")

        self.root.configure(
            bg="#1E1E2E"
        )

        self.pipeline = None
        self.accuracy = 0.0
        self.cm = None
        self.canvas = None

        self.setup_ui()

        self.train_ml_model()

    # ========================================================
    # GUI
    # ========================================================

    def setup_ui(self):

        # Header
        title = tk.Label(
            self.root,
            text="📧 Phishing Email Detection Model",
            font=("Segoe UI", 16, "bold"),
            fg="#F38BA8",
            bg="#1E1E2E"
        )

        title.pack(
            pady=(15, 5)
        )

        # Notebook
        self.notebook = ttk.Notebook(
            self.root
        )

        self.notebook.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        # ====================================================
        # TAB 1 - LIVE PREDICTOR
        # ====================================================

        self.tab_predict = tk.Frame(
            self.notebook,
            bg="#1E1E2E"
        )

        self.notebook.add(
            self.tab_predict,
            text=" Live Predictor "
        )

        tk.Label(
            self.tab_predict,
            text="Enter Email Body / Text to Analyze:",
            font=("Segoe UI", 10, "bold"),
            fg="#CDD6F4",
            bg="#1E1E2E"
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 5)
        )

        self.email_input = scrolledtext.ScrolledText(
            self.tab_predict,
            height=7,
            font=("Segoe UI", 10),
            bg="#313244",
            fg="#CDD6F4",
            insertbackground="#CDD6F4",
            relief="flat"
        )

        self.email_input.pack(
            fill="x",
            padx=20,
            pady=5
        )

        # Default example
        self.email_input.insert(
            tk.END,
            "URGENT: Your account access is restricted! "
            "Verify details now at "
            "http://secure-bank-verify.com"
        )

        # Buttons
        btn_frame = tk.Frame(
            self.tab_predict,
            bg="#1E1E2E"
        )

        btn_frame.pack(
            pady=10
        )

        tk.Button(
            btn_frame,
            text="Classify Email",
            command=self.classify_email,
            font=("Segoe UI", 10, "bold"),
            bg="#89B4FA",
            fg="#11111B",
            relief="flat",
            padx=15,
            pady=5,
            cursor="hand2"
        ).pack(
            side="left",
            padx=5
        )

        tk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_email,
            font=("Segoe UI", 10),
            bg="#45475A",
            fg="#CDD6F4",
            relief="flat",
            padx=10,
            pady=5
        ).pack(
            side="left",
            padx=5
        )

        # Result Box
        self.result_box = tk.LabelFrame(
            self.tab_predict,
            text=" Prediction Output ",
            font=("Segoe UI", 10, "bold"),
            fg="#89B4FA",
            bg="#1E1E2E",
            bd=1
        )

        self.result_box.pack(
            fill="x",
            padx=20,
            pady=10
        )

        self.prediction_lbl = tk.Label(
            self.result_box,
            text="Classification: Waiting for analysis...",
            font=("Segoe UI", 12, "bold"),
            fg="#A6ADC8",
            bg="#1E1E2E"
        )

        self.prediction_lbl.pack(
            pady=10
        )

        self.confidence_lbl = tk.Label(
            self.result_box,
            text="Model Confidence Score: N/A",
            font=("Segoe UI", 10, "italic"),
            fg="#BAC2DE",
            bg="#1E1E2E"
        )

        self.confidence_lbl.pack(
            pady=(0, 10)
        )

        # ====================================================
        # TAB 2 - MODEL EVALUATION
        # ====================================================

        self.tab_metrics = tk.Frame(
            self.notebook,
            bg="#1E1E2E"
        )

        self.notebook.add(
            self.tab_metrics,
            text=" Model Evaluation "
        )

        self.acc_label = tk.Label(
            self.tab_metrics,
            text="Model Testing Accuracy: Loading...",
            font=("Segoe UI", 11, "bold"),
            fg="#A6E3A1",
            bg="#1E1E2E"
        )

        self.acc_label.pack(
            pady=10
        )

        self.plot_frame = tk.Frame(
            self.tab_metrics,
            bg="#1E1E2E"
        )

        self.plot_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

    # ========================================================
    # 3. TRAIN MACHINE LEARNING MODEL
    # ========================================================

    def train_ml_model(self):

        X, y = get_dataset()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )

        # TF-IDF + Naive Bayes
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=True
                )
            ),
            (
                "classifier",
                MultinomialNB()
            )
        ])

        # Train
        self.pipeline.fit(
            X_train,
            y_train
        )

        # Predict test data
        y_pred = self.pipeline.predict(
            X_test
        )

        # Accuracy
        self.accuracy = (
            accuracy_score(
                y_test,
                y_pred
            ) * 100
        )

        # Confusion Matrix
        self.cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[
                "Phishing",
                "Safe"
            ]
        )

        # Update GUI
        self.acc_label.config(
            text=f"Model Testing Accuracy: {self.accuracy:.1f}%"
        )

        self.draw_confusion_matrix()

    # ========================================================
    # 4. CONFUSION MATRIX
    # ========================================================

    def draw_confusion_matrix(self):

        fig, ax = plt.subplots(
            figsize=(4.5, 3.5),
            dpi=100
        )

        fig.patch.set_facecolor(
            "#1E1E2E"
        )

        ax.set_facecolor(
            "#1E1E2E"
        )

        # Matrix
        image = ax.imshow(
            self.cm,
            cmap="Blues"
        )

        # Axis labels
        labels = [
            "Phishing",
            "Safe"
        ]

        ax.set_xticks(
            range(len(labels))
        )

        ax.set_yticks(
            range(len(labels))
        )

        ax.set_xticklabels(
            labels,
            color="#CDD6F4"
        )

        ax.set_yticklabels(
            labels,
            color="#CDD6F4"
        )

        ax.set_xlabel(
            "Predicted Label",
            color="#CDD6F4",
            fontweight="bold"
        )

        ax.set_ylabel(
            "True Label",
            color="#CDD6F4",
            fontweight="bold"
        )

        ax.set_title(
            "Confusion Matrix",
            color="#89B4FA",
            pad=12
        )

        # Add numbers
        for i in range(
            self.cm.shape[0]
        ):

            for j in range(
                self.cm.shape[1]
            ):

                value = self.cm[i, j]

                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value < 2 else "black",
                    fontweight="bold"
                )

        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(
            fig,
            master=self.plot_frame
        )

        self.canvas.draw()

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        plt.close(fig)

    # ========================================================
    # 5. EMAIL CLASSIFICATION
    # ========================================================

    def classify_email(self):

        text = self.email_input.get(
            "1.0",
            tk.END
        ).strip()

        if not text:

            messagebox.showwarning(
                "Warning",
                "Please enter email text to analyze."
            )

            return

        # Prediction
        prediction = self.pipeline.predict(
            [text]
        )[0]

        # Probability
        probabilities = self.pipeline.predict_proba(
            [text]
        )[0]

        confidence = max(
            probabilities
        ) * 100

        # Result
        if prediction == "Phishing":

            self.prediction_lbl.config(
                text="⚠️ Result: PHISHING EMAIL DETECTED",
                fg="#F38BA8"
            )

        else:

            self.prediction_lbl.config(
                text="✔ Result: SAFE EMAIL",
                fg="#A6E3A1"
            )

        self.confidence_lbl.config(
            text=f"Model Confidence Score: {confidence:.2f}%"
        )

    # ========================================================
    # 6. CLEAR BUTTON
    # ========================================================

    def clear_email(self):

        self.email_input.delete(
            "1.0",
            tk.END
        )

        self.prediction_lbl.config(
            text="Classification: Waiting for analysis...",
            fg="#A6ADC8"
        )

        self.confidence_lbl.config(
            text="Model Confidence Score: N/A"
        )


# ============================================================
# 7. APPLICATION START
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = PhishingDetectorApp(
        root
    )

    root.mainloop()