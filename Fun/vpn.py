from nordvpn_connect import initialize_vpn, rotate_VPN, close_vpn_connection
import time
settings = initialize_vpn("Miami")  # starts nordvpn and stuff
rotate_VPN(settings)  # actually connect to server

# YOUR STUFF
#time.sleep(10)  # simulate doing stuff for 10 seconds

#close_vpn_connection(settings)

