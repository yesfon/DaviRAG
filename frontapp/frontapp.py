import base64
import streamlit as st
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage


class FrontApp:
    def __init__(self, rag_agent):
        self._load_assets()
        self._setup_interface()
        self.rag_agent = rag_agent
        self._custom_css()
        self._initialize_session()

    def _load_assets(self):
        icon_path = Path("icon") / "logo_davivienda.png"
        self.icon_base64 = self._img_to_base64(icon_path)

    def _img_to_base64(self, image_path: Path) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def _custom_css(self):
        st.markdown(f"""
        <style>
            .header-container {{
                text-align: center;
                padding: 2rem 1rem;
                background: #ffffff;
                border-bottom: 3px solid #cc0000;
                margin-bottom: 2rem;
            }}

            .title-text {{
                color: #cc0000;
                font-size: 2.5em;
                font-family: 'Arial Black', sans-serif;
                margin: 0.5rem 0;
            }}
        </style>
        """, unsafe_allow_html=True)

    def _initialize_session(self):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def _setup_interface(self):
        st.set_page_config(
            page_title="Asistente Corporativo",
            page_icon="icon/logo_davivienda.png",
            layout="centered"
        )
        with st.container():
            st.markdown(
                f"""
                <div class="header-container">
                    <img src="data:image/png;base64,{self.icon_base64}" 
                         width="150" 
                         style="margin-bottom: 1rem;">
                    <div class="title-text">Asistente Inteligente</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        self.chat_container = st.container()

    def display_chat_history(self):
        for message in st.session_state.chat_history:
            role = "Human" if isinstance(message, HumanMessage) else "AI"
            with st.chat_message(role):
                st.markdown(message.content)

    def process_user_input(self):
        user_query = st.chat_input("Your message")
        if user_query:

            st.session_state.chat_history.append(HumanMessage(user_query))
            with st.chat_message("Human"):
                st.markdown(user_query)


            with st.chat_message("AI"):
                ai_response = self.rag_agent.run(user_query)
                st.markdown(ai_response)

            # Guardar respuesta en historial
            st.session_state.chat_history.append(AIMessage(ai_response))

    def run(self):
        self.display_chat_history()
        self.process_user_input()