#!/usr/bin/env python3
"""
Quick fix for Streamlit caching error
This script patches the agribot_streamlit.py file to fix the unhashable parameter error
"""

import re
from pathlib import Path

def fix_cache_decorator():
    """Fix the @st.cache_resource decorator issue"""
    
    file_path = Path("agribot_streamlit.py")
    
    if not file_path.exists():
        print("❌ agribot_streamlit.py not found!")
        return False
    
    print("📝 Reading file...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already using session_state approach
    if 'if \'agent\' not in st.session_state:' in content and '@st.cache_resource' not in content.split('def initialize_agribot')[0] if 'def initialize_agribot' in content else True:
        print("✅ File already fixed!")
        return True
    
    # Backup original
    backup_path = Path("agribot_streamlit.py.backup")
    print(f"💾 Creating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Find and replace the problematic section
    old_pattern = r'@st\.cache_resource\s+def initialize_agribot\(\):'
    
    if re.search(old_pattern, content):
        # Remove @st.cache_resource from initialize_agribot
        content = re.sub(old_pattern, 'def initialize_agribot():', content)
        print("✅ Removed @st.cache_resource decorator")
    else:
        print("⚠️  Pattern not found, manual fix may be needed")
    
    # Write fixed content
    print("💾 Writing fixed file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ File fixed successfully!")
    return True

def add_simple_caching():
    """Add simple session-based caching instead"""
    
    file_path = Path("agribot_streamlit.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if session state init exists
    if 'if \'agent\' not in st.session_state:' in content:
        print("✅ Session state caching already in place")
        return True
    
    # Find where to add it
    init_pattern = r'st\.session_state\.agent = initialize_agribot\(\)'
    
    if re.search(init_pattern, content):
        replacement = """if 'agent' not in st.session_state:
    with st.spinner("🌾 Initializing AgriBot..."):
        st.session_state.agent = initialize_agribot()"""
        
        content = re.sub(init_pattern, replacement, content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Added session state caching")
        return True
    else:
        print("⚠️  Could not find initialization pattern")
        return False

def main():
    print("="*70)
    print("🔧 AgriBot Cache Error Fix")
    print("="*70)
    print()
    
    print("This script will fix the 'UnhashableParamError' in agribot_streamlit.py")
    print()
    
    # Step 1: Fix decorator
    print("Step 1: Fixing @st.cache_resource decorator...")
    if fix_cache_decorator():
        print()
    else:
        print("❌ Fix failed!")
        return
    
    # Step 2: Add simple caching
    print("Step 2: Adding session state caching...")
    if add_simple_caching():
        print()
    else:
        print("⚠️  Partial fix - may need manual adjustment")
    
    print("="*70)
    print("✅ Fix Complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Review the changes in agribot_streamlit.py")
    print("2. Run: streamlit run agribot_streamlit.py")
    print("3. If issues persist, restore backup: agribot_streamlit.py.backup")
    print()
    print("The fix removes problematic caching and uses session state instead.")
    print()

if __name__ == "__main__":
    main()