import serial
import time

class MightyZapL12:
    def __init__(self, port='COM4', baudrate=57600):
        self.ser = serial.Serial(port, baudrate, timeout=0.2)
        self.ID = 0  # Default ID

    def _send_packet(self, command, factors):
        # Format: 3xHdr + ID + SIZE + CMD + Factors + Checksum
        size = 1 + len(factors) + 1 
        packet_body = [self.ID, size, command] + factors
        checksum = (~(sum(packet_body) & 0xFF)) & 0xFF
        packet = bytearray([0xFF, 0xFF, 0xFF] + packet_body + [checksum])
        self.ser.write(packet)
        return packet

    def _read_raw(self, addr, length):
        """Internal reader: Returns (error_byte, data_bytes)"""
        self.ser.reset_input_buffer()
        self._send_packet(0xF2, [addr, length]) # 0xF2 = Load Data
        time.sleep(0.06) # Required processing time for MightyZap
        response = self.ser.read(self.ser.in_waiting)
        
        idx = response.rfind(b'\xff\xff\xff')
        if idx != -1:
            data_segment = response[idx:]
            # Structure: Header(3) + ID(1) + Size(1) + Error(1) + Data(N) + CS(1)
            if len(data_segment) >= (7 + length):
                error_byte = data_segment[5]
                payload = data_segment[6 : 6 + length]
                return error_byte, payload
        return None, None

    # --- ALARM & DIAGNOSTICS ---
    def check_errors(self):
        """Queries the status and prints active alarms."""
        # Querying voltage is a safe way to trigger a status return
        err_byte, _ = self._read_raw(0x92, 1)
        
        if err_byte is None:
            print("❌ Communication Error: No response from actuator.")
            return False

        if err_byte == 0:
            print("✅ System Status: OK")
            return True

        errors = []
        if err_byte & 0x01: errors.append("Input Voltage Error")
        if err_byte & 0x04: errors.append("Overheating Error")
        if err_byte & 0x08: errors.append("Range Error")
        if err_byte & 0x20: errors.append("Overload Error")
        if err_byte & 0x40: errors.append("Instruction Error")
        
        print(f"⚠️ ALARMS DETECTED: {', '.join(errors)}")
        return False

    # --- SETTER FUNCTIONS ---
    def set_position(self, position):
        """Sets target position (0-4095) | Reg 0x86"""
        return self._send_packet(0xF3, [0x86, position & 0xFF, (position >> 8) & 0xFF])

    def set_speed(self, rate):
        """Sets speed (1-255) | Reg 0x21 & 0x22"""
        self._send_packet(0xF3, [0x21, rate])
        time.sleep(0.02)
        return self._send_packet(0xF3, [0x22, rate])

    def set_force(self, on_off):
        """Motor Torque On (1) or Off (0) | Reg 0x80"""
        return self._send_packet(0xF3, [0x80, 1 if on_off else 0])

    # --- GETTER FUNCTIONS ---
    def get_position(self):
        _, res = self._read_raw(0x8C, 2)
        return (res[1] << 8) | res[0] if res else None

    def get_speed(self):
        _, res = self._read_raw(0x8E, 2)
        return (res[1] << 8) | res[0] if res else None

    def is_moving(self):
        _, res = self._read_raw(0x96, 1)
        return res[0] == 1 if res else False

    def get_voltage(self):
        _, res = self._read_raw(0x92, 1)
        return res[0] / 10.0 if res else None

    def get_temperature(self):
        _, res = self._read_raw(0x13, 1)
        return int(res[0]) if res else None

    # --- MAINTENANCE ---
    def restart(self):
        print("Sending Restart command...")
        return self._send_packet(0xF8, [])

    def factory_reset(self):
        print("Performing Factory Reset...")
        return self._send_packet(0xF6, [])
    
    def clear_alarm(self):
        print("delete error")
        self.set_force(False)  
        time.sleep(0.5)
        self.restart()        
        time.sleep(0.5)
        self.set_force(True)   
        print("System ready.")

# In[]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure you use the correct port for your system
    mz = MightyZapL12('COM4')

    print("Initializing...")
    mz.set_force(True)
    
    positions = [360]
    speeds = [8]

    try:
        for target_pos, move_speed in zip(positions, speeds):
            print(f"\n--- Moving to {target_pos} (Speed: {move_speed}) ---")
            mz.set_speed(move_speed)
            mz.set_position(target_pos)
            
            # Monitor Movement
            while mz.is_moving():
                pos = mz.get_position()
                spd = mz.get_speed()
                print(f"Current Position: {pos} | Current Speed: {spd}", end='\r')
                time.sleep(0.01)
            
            print(f"\nMotion Complete. Final Position: {mz.get_position()}")
            
            # --- POST-MOVEMENT DIAGNOSTIC ---
            print("Checking System Health...")
            mz.check_errors()
            print(f"Voltage: {mz.get_voltage()}V | Temp: {mz.get_temperature()}°C")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user.") 
