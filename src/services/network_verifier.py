import socket
import requests

import netifaces

from ..utils.logger_utils import get_logger

logger = get_logger(__name__)

class NetworkUtils:
    @staticmethod
    def get_local_machine_ip():
        ip_addresses = []

        # Method 1: Using socket
        try:
            socket_ip = socket.gethostbyname(socket.gethostname())
            if socket_ip != "127.0.0.1":
                ip_addresses.append(("socket", socket_ip))
            logger.info(f"[SOCKET] IP address: {socket_ip}")
        except Exception as e:
            logger.error(f"Failed to fetch IP address using socket: {e}")

        # Method 2: Using netifaces (cross-platform)
        try:
            for interface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        ip = addr["addr"]
                        if ip != "127.0.0.1":
                            ip_addresses.append(("netifaces", ip))
                            logger.info(f"[NETIFACES] IP address for {interface}: {ip}")
        except Exception as e:
            logger.error(f"Failed to fetch IP addresses using netifaces: {e}")

        # Analyze results
        if not ip_addresses:
            logger.warning("No non-loopback IP addresses found.")
            return None
        elif len(ip_addresses) == 1:
            logger.info(f"Found one IP address: {ip_addresses[0][1]}")
            return ip_addresses[0][1]
        else:
            logger.info(
                f"Found multiple IP addresses: {[ip for _, ip in ip_addresses]}"
            )
            return ip_addresses[0][1]  # Returning the first non-loopback IP found

    @staticmethod
    def get_external_machine_ip():
        try:
            response = requests.get("https://api.ipify.org")
            response.raise_for_status()
            external_ip = response.text
            logger.info(f"External IP address: {external_ip}")
            return external_ip
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch external IP address: {e}")
            return None