import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="YouTube Script Writer", layout="wide")
st.title("🎬 YouTube Script Writer")
st.write("Create engaging YouTube scripts with AI-powered research and writing!")

# Initialize tools and models
@st.cache_resource
def initialize_components():
    search_tool = DuckDuckGoSearchRun()
    llm = ChatOllama(model="llama3.2:1b")
    return search_tool, llm

search_tool, llm = initialize_components()

# User inputs
st.write("### 📝 Script Requirements")
topic = st.text_input("What's your video topic?", placeholder="e.g., How to make the perfect pizza at home")

col1, col2 = st.columns(2)
with col1:
    video_length = st.number_input("Video length (minutes)", min_value=1, max_value=60, value=10)
    
with col2:
    creativity = st.slider("Creativity Level", min_value=0.0, max_value=1.0, value=0.7, 
                          help="Lower = more factual, Higher = more creative")

# Script generation function
def generate_script(topic, video_length, creativity):
    # Research the topic
    st.info("🔍 Researching your topic...")
    research_query = f"latest information about {topic} 2024"
    research_results = search_tool.run(research_query)
    
    # Create script prompt
    script_prompt = PromptTemplate(
        input_variables=["topic", "video_length", "research", "creativity"],
        template="""
You are a professional YouTube script writer. Create an engaging script for a {video_length}-minute video about "{topic}".

Research information to include:
{research}

Guidelines:
- Hook viewers in the first 30 seconds
- Include engaging transitions and call-to-actions
- Make it conversational and easy to follow
- Include timestamps for key sections
- End with a strong conclusion and call-to-action
- Creativity level: {creativity} (0=factual, 1=very creative)

Format the script with clear sections and timestamps. Make it engaging and informative!
        """
    )
    
    # Generate script
    st.info("✍️ Writing your script...")
    
    # Set temperature based on creativity slider
    llm.temperature = creativity
    
    # Use modern RunnableSequence approach
    chain = script_prompt | llm
    
    script_msg = chain.invoke({
        "topic": topic,
        "video_length": video_length,
        "research": research_results,
        "creativity": creativity
    })
    script = script_msg.content
    
    return script, research_results

# Generate button
if st.button("🎬 Generate Script", type="primary"):
    if topic:
        with st.spinner("Creating your YouTube script..."):
            try:
                script, research = generate_script(topic, video_length, creativity)
                
                # Display results
                st.success("✅ Script generated successfully!")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📄 Full Script", "🔍 Research", "📊 Script Analysis"])
                
                with tab1:
                    st.subheader("🎬 Your YouTube Script")
                    st.text_area("Script", script, height=600, key="script_output")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Script",
                        data=script,
                        file_name=f"youtube_script_{topic.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                
                with tab2:
                    st.subheader("🔍 Research Results")
                    st.write(research)
                
                with tab3:
                    st.subheader("📊 Script Analysis")
                    word_count = len(script.split())
                    estimated_time = word_count / 150  # Average speaking rate
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Word Count", word_count)
                    with col2:
                        st.metric("Estimated Duration", f"{estimated_time:.1f} min")
                    with col3:
                        st.metric("Creativity Level", f"{creativity:.1f}")
                    
                    # Script structure analysis
                    st.write("**Script Structure:**")
                    if "hook" in script.lower() or "introduction" in script.lower():
                        st.write("✅ Strong opening detected")
                    if "conclusion" in script.lower() or "call to action" in script.lower():
                        st.write("✅ Proper conclusion with CTA")
                    if any(word in script.lower() for word in ["first", "second", "next", "finally"]):
                        st.write("✅ Good transitions and structure")
                
            except Exception as e:
                st.error(f"❌ Error generating script: {e}")
                st.info("💡 Try adjusting the topic or creativity level and try again.")
    else:
        st.warning("⚠️ Please enter a topic for your video script.")

# Tips and examples
with st.expander("💡 Tips for Better Scripts"):
    st.write("""
    **Best Practices:**
    - Be specific with your topic (e.g., 'How to make Neapolitan pizza' vs 'How to cook')
    - Use creativity level 0.3-0.5 for educational content
    - Use creativity level 0.7-0.9 for entertainment content
    - Keep video length realistic for your topic
    
    **Example Topics:**
    - "10 Life-Changing Productivity Hacks for 2024"
    - "The Complete Guide to Starting a YouTube Channel"
    - "5 Easy Recipes for College Students"
    - "How to Master Public Speaking in 10 Minutes"
    """)

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain, Ollama, and DuckDuckGo Search*")
