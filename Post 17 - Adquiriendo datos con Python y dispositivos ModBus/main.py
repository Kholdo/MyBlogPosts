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

if connection:
    # we read first the integer data
    for key, value in step_th()['memo_Integers'].items():
        rr = client.read_holding_registers(value, 1, unit=0x01)
        if not rr.isError():
            val = rr.registers[0]
            print('{}: {}'.format(key, val))
        else:
            print('{}: error'.format(key))

    for key, value in step_th()['memo_Floats'].items():
        rr1 = client.read_holding_registers(value, 1, unit=0x01)
        rr2 = client.read_holding_registers(value, 2, unit=0x01)
        if not rr1.isError() and not rr2.isError():
            print('{}: {};{}'.format(key, rr1.registers[0], rr2.registers[0]))
            bin_number = '0' + str(np.base_repr(rr2.registers[0], base=2)) \
                         + '0' + str(np.base_repr(rr1.registers[0], base=2))
            f = int(bin_number, 2)
            print('{}:{}'.format(key, struct.unpack('f', struct.pack('I', f))[0]))
        else:
            print('{}: error'.format(key))

    client.close()