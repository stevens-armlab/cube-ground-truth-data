See in the cube surface points folder.

Only data under label Q0, Qx, Qy, Qz , Tx, Ty, Tz are useful, discard other data.

Suppose there is a quaternion *q = ax + by + cz + w*

Q0=w

Qx=a

Qy=b

Qz=c

Tx ,Ty, Tz stands for the position of the probe's center in the sensor coordinate system

We are using the Aurora 5DOF tracker, so the Qz will always be zero and it has nothing to do with our result. 

For more sensor information, see in this link: https://www.ndigital.com/electromagnetic-tracking-technology/aurora/



Set up NDI tracker:

* To get *Aurora* NDI track software you need to register an account by contacting customer service of *Aurora*

* After getting the account download NDI Toolbox through: https://www.ndigital.com/products/software/

* For **Windows** you need install USB driver for NDI tracker following these steps:

1. From the Windows **Start** menu, select **Control Panel**.
2. In the top right hand corner of the Control Panel window, select **View by: Small icons**.
3. From the Control panel, select **Device Manager**.
4. Under **Other Devices**, right click **NDI Aurora SCU**.
5. Select **Update Driver Software...**, then select **Browse my computer for driver software**.
6. Select **Program Files (x86)\Northern Digital Inc\ToolBox\USB Driver**. Select **Next**.
7. In the Windows security dialog, select **Install this driver software anyway** option. The driver will install.

* For **Linux**: 

  On Linux kernel versions 2.6 and later, USB serial devices appear as driver files "/dev/ttyUSBx" (where x is the port number). These drivers emulate a standard tty serial port and allow applications to communicate through the USB device as if it were an RS-232 or RS-422 port.

  For Linux kernel versions 2.6.8 through 2.6.30, NDI has supplied patches to allow the kernel to recognize and configure the Aurora System. (The patch files are located in the<ToolBox_install_dir>/usb-patch/ directory after ToolBox has been installed.) Kernel versions 2.6.32 and later automatically recognize the Aurora System and no patching is required.

* Then open the NDI Track application, under **File** option select **Load Virtulal SROM**. Load the ROM file that matches the NDI tracker sensor you are using. If you are using the same sensor as us you can Load the ROM file *DDRO-080-051-01_GENERIC.rom*.

* Great you are ready to collect data using the NDI Trakcer!

