gmx trjconv -s nvt_auto.tpr -f traj_first1000 -o traj_first1000_unwrapped.xtc -pbc nojump


echo "0" | gmx msd -f traj_first1000_unwrapped.xtc -s nvt.tpr -n index.ndx -o MSD_Z.xvg -lateral z

