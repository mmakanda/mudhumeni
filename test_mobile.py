#!/usr/bin/env python3
"""
Quick Mobile Testing Helper
Generates QR code and shows how to test on mobile devices
"""

import socket
import qrcode
from io import BytesIO
import base64

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def generate_qr_code(url):
    """Generate QR code for easy mobile access"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save to file
    img.save("mobile_access_qr.png")
    print("✅ QR code saved as: mobile_access_qr.png")
    
    return img

def main():
    print("="*70)
    print("📱 AgriBot Pro - Mobile Testing Helper")
    print("="*70)
    print()
    
    # Get local IP
    local_ip = get_local_ip()
    port = 8501
    
    # URLs
    local_url = f"http://localhost:{port}"
    network_url = f"http://{local_ip}:{port}"
    
    print("🌐 Access URLs:")
    print(f"   Local:   {local_url}")
    print(f"   Network: {network_url}")
    print()
    
    print("📱 Testing on Mobile Devices:")
    print()
    print("Option 1: Same WiFi Network")
    print(f"   1. Connect your phone to the same WiFi")
    print(f"   2. Open browser and go to: {network_url}")
    print()
    
    print("Option 2: QR Code (Easiest)")
    try:
        generate_qr_code(network_url)
        print(f"   1. Open mobile_access_qr.png")
        print(f"   2. Scan with your phone camera")
        print(f"   3. Tap the link")
        print()
    except Exception as e:
        print(f"   ⚠️ QR generation failed: {e}")
        print(f"   Install qrcode: pip install qrcode[pil]")
        print()
    
    print("Option 3: Ngrok (Public Internet)")
    print("   1. Install ngrok: https://ngrok.com")
    print("   2. Run: ngrok http 8501")
    print("   3. Share the ngrok URL with anyone")
    print()
    
    print("="*70)
    print("🧪 Testing Checklist:")
    print("="*70)
    print()
    print("Test on these devices:")
    print("   [ ] iPhone (Safari)")
    print("   [ ] Android (Chrome)")
    print("   [ ] iPad (Safari)")
    print("   [ ] Android Tablet")
    print()
    print("Check these features:")
    print("   [ ] Home page loads")
    print("   [ ] Navigation works")
    print("   [ ] Chat works")
    print("   [ ] Forms work")
    print("   [ ] Buttons tap correctly")
    print("   [ ] Text is readable")
    print("   [ ] Images display")
    print()
    print("Test orientations:")
    print("   [ ] Portrait mode")
    print("   [ ] Landscape mode")
    print()
    
    print("="*70)
    print("💡 Tips:")
    print("="*70)
    print()
    print("1. Desktop browser simulation:")
    print("   - Chrome: F12 → Toggle device toolbar")
    print("   - Test different screen sizes")
    print()
    print("2. Real device testing:")
    print("   - Best for accurate results")
    print("   - Test touch interactions")
    print("   - Check keyboard behavior")
    print()
    print("3. Performance testing:")
    print("   - Check load times")
    print("   - Monitor data usage")
    print("   - Test on slow connections")
    print()
    
    print("="*70)
    print("🚀 Ready to Test!")
    print("="*70)
    print()
    print("Start your app with: streamlit run agribot_streamlit.py")
    print(f"Then access on mobile: {network_url}")
    print()

if __name__ == "__main__":
    main()