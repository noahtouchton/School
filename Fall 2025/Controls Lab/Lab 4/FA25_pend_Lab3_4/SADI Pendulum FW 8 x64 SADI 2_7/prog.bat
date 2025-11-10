::readme
::1) To program SADI make sure that the slide switch is toggled to where the boot LED is illuminated
::	Either unplug/replug the USB connection or press the MCU reset button to put the microcontroller into boot mode
::2) Open prog.bat to program the firmware
::
::Changing Firmware
::To change the firmware change the file name after -U
::Include the new firmware file in this directory 

avrdude -p ATxmega64A3U -c flip2 -e -U SADI_Pendulum.hex
TIMEOUT /T 15