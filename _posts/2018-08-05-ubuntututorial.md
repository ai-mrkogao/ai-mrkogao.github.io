---
title: "Ubuntu tutorial(pycharm,eclipse,vim,java,pyqt4 and etc)"
date: 2018-08-05
classes: wide
use_math: true
tags: ubuntu command pycharm eclipse java
category: ubuntu
---

# open-vm-tools and VMWare Shared Folders for Ubuntu guests 
[open-vm-tools and VMWare Shared Folders for Ubuntu guests ](https://gist.github.com/darrenpmeyer/b69242a45197901f17bfe06e78f4dee3)

```bash
sudo mount -t fuse.vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
```

# Pycharm theme change
![](../../pictures/ubuntu/pycharmthemechange.png){:height="80%" width="80%"}

# Using Vim Editor Emulation in PyCharm (IdeaVim)
[Using Vim Editor Emulation in PyCharm (IdeaVim)](https://www.jetbrains.com/help/pycharm/using-product-as-the-vim-editor.html)


# How To Install Java with Apt-Get on Ubuntu 16.04 
[How To Install Java with Apt-Get on Ubuntu 16.04 ](https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04)

# How to install PyQt5 in Python 3 (Ubuntu 14.04)
[How to install PyQt5 in Python 3 (Ubuntu 14.04)](https://stackoverflow.com/questions/36757752/how-to-install-pyqt5-in-python-3-ubuntu-14-04)

# How can I empty the trash using terminal?
[How can I empty the trash using terminal?](https://askubuntu.com/questions/468721/how-can-i-empty-the-trash-using-terminal)

```bash
sudo rm -rf ~/.local/share/Trash/*
```

# How do I mount shared folders in Ubuntu using VMware tools?
[How do I mount shared folders in Ubuntu using VMware tools?](https://askubuntu.com/questions/29284/how-do-i-mount-shared-folders-in-ubuntu-using-vmware-tools)

# VirtualBox/SharedFolders
[VirtualBox/SharedFolders](https://help.ubuntu.com/community/VirtualBox/SharedFolders)

# Mounting Shared Folders in a Linux Guest
[Mounting Shared Folders in a Linux Guest](https://pubs.vmware.com/workstation-9/index.jsp?topic=%2Fcom.vmware.ws.using.doc%2FGUID-AB5C80FE-9B8A-4899-8186-3DB8201B1758.html)

```bash
sudo mount -t vmhgfs .host:/ushare /home/home9/Downloads
sudo mount -t vmhgfs .host:/home/home9/Downloads /ushare
```
# Access VMware Workstation Host Folders from Ubuntu 17.10 Guest Machines
[Access VMware Workstation Host Folders from Ubuntu 17.10 Guest Machines](https://websiteforstudents.com/access-vmware-host-folders-guest-machine-ubuntu-17-10/)

```bash
sudo /tmp/vmware-tools-distrib/vmware-install.pl -d
```


# unable to see shared folders in ubuntu(guest) installed in vmware
[unable to see shared folders in ubuntu(guest) installed in vmware](https://askubuntu.com/questions/331671/unable-to-see-shared-folders-in-ubuntuguest-installed-in-vmware)


```bash
sudo vmware-config-tools.pl
sudo ./vmware-intsll.pl
chmod o+rx /mnt
chmod o+rx /mnt/hgfs/
chmod o+rx /mnt/hgfs/foldername 
```

