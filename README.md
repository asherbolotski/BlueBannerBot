# 🤖 Blue Banner Bot
# https://asherbolotski.github.io/BlueBannerBot

## What is it?

**Blue Banner Bot** is an intelligent tool designed to answer your questions about the FIRST Robotics Competition (FRC). Built for students, mentors, and volunteers, it serves as a quick and reliable resource for all things FRC.

You can ask technical questions about specific components, rules, or programming libraries. For example:

- *How do I wire a Talon FX motor controller?*
- *What is the maximum height for a robot this season?*
- *Can you show me an example of how to use PID control in WPILib?*
- *What's the difference between a NEO and a Falcon 500?*

---

## 📚 Sources of Information

To provide accurate and relevant answers, the bot pulls information from a curated list of trusted FRC resources, including:

- **[WPILib Docs](https://docs.wpilib.org/):** The official documentation for the FRC control system software library.
- **[WPILib Java Docs](https://github.wpilib.org/allwpilib/docs/release/java/index.html):** The official Java API documentation for FRC libraries.
- **[REV Robotics](https://docs.revrobotics.com/):** Documentation for REV hardware and software products like the NEO motors, Spark MAX controllers, and more.
- **[CTRE Phoenix](https://docs.ctre-phoenix.com/):** Documentation for Cross The Road Electronics products, including the Talon motor controllers and CANivore.
- **[AndyMark](https://www.andymark.com/):** Product specifications and guides for various mechanical and electrical components.

---

## 🛠️ How It's Implemented

The bot is built with a modern tech stack to deliver a seamless and intelligent experience.

### **Frontend**

The user interface is a clean, single-page application built with **HTML**, **Tailwind CSS**, and **vanilla JavaScript**. It's designed to be simple and intuitive, allowing you to ask questions and receive answers quickly in a familiar chat format.

### **Backend**

The magic happens on the backend, powered by a **Python API using FastAPI**. Technologies used include:

- **OpenAI:** Uses the `gpt-5-mini` model to comprehend your questions and formulate human-like answers. For document searching, it uses the `text-embedding-3-small` model.
- **Pinecone:** A vector database that efficiently stores and searches through vast amounts of FRC documentation. The bot uses a hybrid search approach, combining dense vectors (semantic meaning) and sparse vectors (keyword matching) to find the most relevant information.

---

## 🚀 Future Features

I'm always working on making Blue Banner Bot even better! Here are some features on the roadmap:

- WPILib C++ docs
- Benchmarking/finetuning against forum responses
- More data sources, such as the game manual, LimeLight, PhotonVision
- User recommended features - use the feedback form below!

---

## 💬 How to Provide Feedback

Got a suggestion, found a bug, or think an answer could be better? I'd love to hear from you!

- **In-App Feedback:** Use the built-in "Feedback" button to send your thoughts directly.
- **Google Form:** You can also fill out a detailed feedback form [here](https://docs.google.com/forms/d/e/1FAIpQLSegvnNhVmj1elVPePl11OM6j4P0xWtqwyh7gKRnopYC6CA0Sg/viewform?usp=sharing&ouid=117254353672431218000).

---

## ❤️ Support the Project

Blue Banner Bot is a passion project, but it incurs real-world costs to operate. If you find the bot useful, please consider supporting its development and maintenance.

Running this service involves costs for:

- **Google Cloud:** Hosting for the backend services.
- **OpenAI API:** Credits for processing and answering questions.
- **Pinecone:** Vector database hosting and operations.

You can leave a tip [here](https://paypal.me/asherbolotski).

---

Thank you for using **Blue Banner Bot**!

- Asher Bolotski, Blue Banner Bot Developer