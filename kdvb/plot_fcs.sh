

plott () {
    # bash main_serial.sh $suffix
    python3 plot_tracker_f.py $suffix/$suffix.cfg
    cp $suffix/approx_f.pdf ~/invmod/sapprox/${suffix}_f.pdf
    cp $suffix/approx_f.png ~/invmod/sapprox/${suffix}_f.png

    # python3 plot_tracker.py $suffix/$suffix.cfg
    # cp $suffix/approx.pdf ~/invmod/approx/$suffix.pdf
    # cp $suffix/approx.png ~/invmod/approx/$suffix.png
}

suffix="C0e1"
plott

suffix="C0"
plott
# exit 1

suffix="C1"
plott

suffix="Cnp004"
plott

cp legend_kdvb.pdf ~/invmod/legend_kdvb.pdf