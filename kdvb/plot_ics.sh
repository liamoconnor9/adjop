

plott () {
    # bash main_serial.sh $suffix
    python3 plot_tracker.py $suffix/$suffix.cfg
    # python3 crop_bottom.py $suffix/approx.pdf
    # exit 1
    # cp $suffix/approx.pdf ~/invmod/approx/$suffix.pdf
    # cp $suffix/approx.png ~/invmod/approx/$suffix.png

    # python3 plot_tracker.py $suffix/$suffix.cfg
    # cp $suffix/approx.pdf ~/invmod/approx/$suffix.pdf
    # cp $suffix/approx.png ~/invmod/approx/$suffix.png
}

suffix="C0"
plott
# exit 1

suffix="C1"
plott

suffix="Cnp004"
plott

suffix="C0ls"
plott

cp legend_kdvb.pdf ~/invmod/legend_kdvb.pdf