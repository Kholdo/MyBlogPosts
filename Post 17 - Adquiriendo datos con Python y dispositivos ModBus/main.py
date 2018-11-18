from pymodbus.client.sync import ModbusSerialClient as ModbusClient #initialize a serial RTU client instance
from stepTHconf import step_th


method = step_th()['default_config']['method']
stopbits = step_th()['default_config']['stopbits']
bytesize = step_th()['default_config']['bytesize']
parity = step_th()['default_config']['parity']
baudrate = step_th()['default_config']['baudrate']
timeout = step_th()['default_config']['timeout']
nper = step_th()['default_config']['nper']
port = '/dev/ttyUSB0'

client = ModbusClient(method=method,
                      stopbits=stopbits,
                      bytesize=bytesize,
                      parity=parity,
                      baudrate=baudrate,
                      timeout=timeout,
                      port=port)

connection = client.connect()