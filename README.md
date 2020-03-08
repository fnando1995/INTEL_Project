# Where should I put my store? - an object tracking system for remote places.


This repro constains a minimal project for a final product to measure outside potential (traffic of person/car/bus/motorcicle/etc...).

For this minimal/demo project there is one video taken from a real store (all credits reserved).

The video is from an external camera (not CCTV) which was created with a raspi 3 B+ (raspi 4 could improve speed), a NCS2 and a usb Camera with at least HD resolution.

### How to run

~~~
git clone https://github.com/fnando1995/INTEL_Project.git
cd INTEL_Project
sudo python3 -m pip install -r req.txt
python3 run.py
~~~

##### note: This project is capable or running in CPU or VPU in case that mounting the rasp+NCS2+CAM takes to much time

soft-nms  https://arxiv.org/pdf/1704.04503.pdf

sort    https://arxiv.org/pdf/1602.00763.pdf

kalman-filters http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf

hungarian-algorithm http://www.or.deis.unibo.it/staff_pages/martello/TechReportEgervary.pdf
