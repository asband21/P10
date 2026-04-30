for i in {1..1000}; do
	sudo pinctrl set 17 op dl
	sleep 0.1
	sudo pinctrl set 17 op dl
	sleep 0.1
done
