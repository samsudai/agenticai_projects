import streamlit as st
from agent import run_support_agent


# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartCare AI Support Agent",
    page_icon="🎧",
    layout="wide"
)


# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.title("🎧 SmartCare AI Support Agent")

st.caption(
    "A LangChain + Streamlit demo that classifies customer issues, checks order data from CSV, "
    "retrieves policy guidance from a PDF, and drafts a professional customer support response."
)


# ---------------------------------------------------------
# Sidebar: Customer Details
# ---------------------------------------------------------
with st.sidebar:
    st.header("Customer Details")

    customer_name = st.text_input(
        "Customer Name",
        value="Priya Sharma"
    )

    order_id = st.text_input(
        "Order ID",
        value="ORD1002",
        help="Try ORD1001, ORD1002, ORD1003, ORD1004, or ORD1005"
    )

    customer_type = st.selectbox(
        "Customer Type",
        ["Regular", "Premium", "Business"],
        index=1
    )

    st.divider()

    st.subheader("Sample Order IDs")

    st.markdown(
        """
- **ORD1001**: Delayed order  
- **ORD1002**: Delivered order  
- **ORD1003**: Returned order with refund pending  
- **ORD1004**: Processing order  
- **ORD1005**: Delivered damaged product  
"""
    )

    st.divider()

    st.subheader("Sample Issues")

    st.markdown(
        """
**Delivery Delay**  
My order was supposed to arrive yesterday, but I still have not received it.

**Missing Package**  
The app says my order was delivered, but I never got it.

**Refund Delay**  
I returned the product 10 days ago, but my refund has not been credited.

**Product Defect**  
The product arrived damaged and is not working.

**Billing Issue**  
I was charged twice for the same order.
"""
    )

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# Initialize Chat History
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Display Existing Chat Messages
# ---------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------
user_message = st.chat_input("Type the customer issue here...")


# ---------------------------------------------------------
# Process Customer Message
# ---------------------------------------------------------
if user_message:
    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_message)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Agent is classifying the issue, checking order data, searching policy PDF, and drafting response..."
            ):
                result = run_support_agent(
                    customer_message=user_message,
                    customer_name=customer_name,
                    customer_type=customer_type,
                    order_id=order_id
                )

            final_answer = result["final_answer"]

            st.markdown(final_answer)

            # Show internal reasoning details for demo purpose
            with st.expander("View Agent Reasoning Details"):
                st.subheader("1. Issue Classification")
                st.json(result["classification"])

                st.subheader("2. Order Data from CSV")
                st.code(result["order_details"], language="json")

                st.subheader("3. Retrieved Policy from PDF")
                st.text(result["policy_text"])

            # Store assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer
                }
            )

        except Exception as e:
            error_message = f"Something went wrong: {str(e)}"

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )