import streamlit as st
import pandas as pd
import re
import plotly.express as px
from dotenv import load_dotenv
from database.db_connector import is_query_safe
from sql_agent.sql_assistant import train_vanna, generate_sql_with_feedback
import os
import json
from uuid import uuid4
import datetime

load_dotenv()

# ---- Datetime Serialization Helper ----
def serialize_datetime(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

# ---- Setup ----
@st.cache_resource
def get_vanna():
    return train_vanna()

vn = get_vanna()

# ---- Feedback Storage ----
FEEDBACK_JSON_PATH = "feedback.json"

def append_feedback_to_json(user_question, incorrect_sql, feedback_text, correct_sql):
    feedback_entry = {
        "id": str(uuid4()),
        "user_question": user_question,
        "incorrect_sql": incorrect_sql,
        "feedback": feedback_text,
        "correct_sql": correct_sql
    }

    if not os.path.exists(FEEDBACK_JSON_PATH):
        with open(FEEDBACK_JSON_PATH, "w") as f:
            json.dump([], f, indent=4)

    with open(FEEDBACK_JSON_PATH, "r+") as f:
        data = json.load(f)
        data.append(feedback_entry)
        f.seek(0)
        json.dump(data, f, indent=4)

# ---- Chat History Management ----
CHAT_HISTORY_DIR = "chat_histories"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

def save_chat_to_file(messages, session_id=None):
    if not session_id:
        session_id = str(uuid4())
        st.session_state.session_id = session_id

    filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    title = next((m["content"] for m in messages if m["role"] == "user"), "Untitled Chat")
    with open(filepath, "w") as f:
        json.dump( {"title": title, "messages": messages}, f, indent=4,default=serialize_datetime)

def load_chat(session_id):
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            return data.get("messages", [])
    return []

def delete_chat(session_id):
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

def list_chats():
    chat_files = [
        os.path.join(CHAT_HISTORY_DIR, f)
        for f in os.listdir(CHAT_HISTORY_DIR)
        if f.endswith(".json")
    ]
    chat_files.sort(key=os.path.getctime, reverse=True)
    chats = []
    for filepath in chat_files[:5]:
        session_id = os.path.basename(filepath).replace(".json", "")
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                title = data.get("title", "Untitled Chat")
                chats.append({"session_id": session_id, "title": title})
        except Exception:
            pass
    return chats

# ---- Utility Functions ----
def user_wants_plot(question: str) -> bool:
    return bool(re.search(r"\b(plot|chart|graph|visualize|draw|visualisation)\b", question.lower()))

def generate_plot_from_df(df):
    if df is None or df.empty:
        st.warning("No data to plot.")
        return

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if categorical_cols and numeric_cols:
        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
    elif len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
    elif numeric_cols:
        fig = px.histogram(df, x=numeric_cols[0])
    else:
        st.warning("Not enough columns to generate a plot.")
        return

    st.plotly_chart(fig, use_container_width=True)

def reset_to_ask_question():
    st.session_state.stage = "ask_question"
    st.session_state.user_question = ""
    st.session_state.sql = ""
    st.session_state.query_result = None
    st.session_state.user_feedback = ""

# ---- Session Initialization ----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = "ask_question"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_question" not in st.session_state:
    st.session_state.user_question = ""
if "sql" not in st.session_state:
    st.session_state.sql = ""
if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = ""

# ---- UI Layout ----
st.title("🤖 SQL Query Assistant")

# ---- Sidebar: Manual SQL ----
st.sidebar.header("Manual SQL Query")
if st.sidebar.button("🆕 New Chat"):
    st.session_state.clear()
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []
    st.rerun()

manual_sql = st.sidebar.text_area("Enter your SQL query here:")
if st.sidebar.button("Run Manual SQL"):
    if is_query_safe(manual_sql):
        try:
            with st.spinner("Executing manual SQL query..."):
                manual_result = vn.run_sql(manual_sql)
            st.success("Manual query executed successfully!")
            st.dataframe(manual_result)
            st.session_state.messages.append({"role": "user", "content": manual_sql})
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Manual SQL Execution",
                "result": manual_result.to_dict(orient="records"),
                "show_plot": user_wants_plot(manual_sql)
            })
            save_chat_to_file(st.session_state.messages, st.session_state.session_id)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.error("Unsafe SQL detected. Only SELECT-like statements allowed.")

# ---- Sidebar: Chat History ----
st.sidebar.header("🗂️ Chat History")
for chat in list_chats():
    with st.sidebar.expander(f"💬 {chat['title'][:60]}"):
        col1, col2 = st.columns([0.7, 0.3])
        if col1.button("Load", key=f"load_{chat['session_id']}"):
            st.session_state.session_id = chat["session_id"]
            st.session_state.messages = load_chat(chat["session_id"])
            st.rerun()
        if col2.button("🗑️", key=f"delete_{chat['session_id']}"):
            delete_chat(chat["session_id"])
            st.rerun()

# ---- Chat Interface ----
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    if msg.get("result") is not None:
        df = pd.DataFrame(msg["result"])
        st.dataframe(df)
        if msg.get("show_plot"):
            with st.expander("📊 View Chart"):
                generate_plot_from_df(df)

# ---- Question/SQL Flow ----
if st.session_state.stage == "ask_question":
    user_input = st.chat_input("Ask a question about your database:")
    if user_input:
        if not is_query_safe(user_input):
            st.error("Error: Queries with DELETE, UPDATE, DROP, INSERT, or other unsafe operations are not allowed.")
        else:
            st.session_state.user_question = user_input
            st.session_state.messages.append({"role": "user", "content": user_input})
            save_chat_to_file(st.session_state.messages, st.session_state.session_id)

            with st.spinner("Generating SQL..."):
                try:
                    sql = generate_sql_with_feedback(
                        vn,
                        user_input,
                        previous_messages=st.session_state.messages
                    )
                    if not sql.strip().lower().startswith(("select", "with", "show", "describe")):
                        st.error(f"Generated text is not valid SQL:\n\n{sql}")
                    else:
                        st.session_state.sql = sql
                        st.session_state.stage = "show_query"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

elif st.session_state.stage == "show_query":
    try:
        result_df = vn.run_sql(st.session_state.sql)
        st.session_state.query_result = result_df
        wants_plot = user_wants_plot(st.session_state.user_question)

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            if st.session_state.messages[-1]["content"] == st.session_state.sql:
                st.session_state.messages.pop()

        st.session_state.messages.append({
            "role": "assistant",
            "content": st.session_state.sql,
            "result": result_df.to_dict(orient="records"),
            "show_plot": wants_plot
        })
        save_chat_to_file(st.session_state.messages, st.session_state.session_id)

        st.session_state.stage = "ask_feedback"
        st.rerun()
    except Exception as e:
        st.error(f"Error executing query: {e}")
        reset_to_ask_question()

elif st.session_state.stage == "ask_feedback":
    st.write("Was this SQL query helpful?")
    col1, col2 = st.columns(2)
    if col1.button("👍 Yes"):
        st.toast("Thanks for the feedback!")
        reset_to_ask_question()
        st.rerun()
    if col2.button("👎 No"):
        st.session_state.stage = "get_feedback"
        st.rerun()

    next_question = st.chat_input("Ask a question about your database:")
    if next_question:
        reset_to_ask_question()
        st.session_state.user_question = next_question
        st.session_state.messages.append({"role": "user", "content": next_question})
        save_chat_to_file(st.session_state.messages, st.session_state.session_id)

        with st.spinner("Generating SQL..."):
            try:
                sql = generate_sql_with_feedback(
                    vn,
                    next_question,
                    previous_messages=st.session_state.messages
                )
                if not sql.strip().lower().startswith(("select", "with", "show", "describe")):
                    st.error(f"Generated text is not valid SQL:\n\n{sql}")
                else:
                    st.session_state.sql = sql
                    st.session_state.stage = "show_query"
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

elif st.session_state.stage == "get_feedback":
    st.subheader("Please provide feedback to improve the SQL query")
    feedback_input = st.text_area("Feedback:", key="user_feedback", placeholder="What needs improvement?")
    if st.button("Submit Feedback"):
        feedback_text = st.session_state.user_feedback.strip()
        if feedback_text:
            with st.spinner("Regenerating SQL..."):
                try:
                    new_sql = generate_sql_with_feedback(vn, st.session_state.user_question, feedback_text)
                    if not new_sql.strip().lower().startswith(("select", "with", "show", "describe")):
                        st.error(f"Invalid regenerated SQL:\n\n{new_sql}")
                    else:
                        # ✅ Save feedback with correct_sql
                        append_feedback_to_json(
                            user_question=st.session_state.user_question,
                            incorrect_sql=st.session_state.sql,
                            feedback_text=feedback_text,
                            correct_sql=new_sql
                        )

                        st.session_state.sql = new_sql
                        st.session_state.messages.append({"role": "user", "content": f"Feedback: {feedback_text}"})
                        st.session_state.messages.append({"role": "assistant", "content": new_sql})
                        save_chat_to_file(st.session_state.messages, st.session_state.session_id)
                        st.session_state.stage = "show_query"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error regenerating SQL: {e}")
        else:
            st.warning("Please provide feedback")
