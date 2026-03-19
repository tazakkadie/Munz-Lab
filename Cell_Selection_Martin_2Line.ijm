name = getTitle();
roiManager("Show All");
roiManager("Reset");
run("Clear Results");

// 1a. Draw Top Curve
showMessage("Draw the top edge of the region.");
done = 0;
while (done == 0) {
	setTool("freeline");
	waitForUser("Please draw line. When complete, press OK.");
	run("Fit Spline");
	Roi.setName("Upper Edge");
	roiManager("Add");
	done = getNumber("If Satisfied, input 1. Otherwise, 0, and redraw.",0);
	if (done == 0){
		roiManager("Select",0);
		roiManager("Delete");
	}	
}

// 1b. Draw Bottom Curve
showMessage("Draw the bottom edge of the region.");
d2 = 0;
while (d2 == 0) {
	setTool("freeline");
	waitForUser("Please draw line. When complete, press OK.");
	run("Fit Spline");
	Roi.setName("Bottom Edge");
	roiManager("Add");
	d2 = getNumber("If Satisfied, input 1. Otherwise, 0, and redraw.",0);
	if (d2 == 0){
		roiManager("Select",1);
		roiManager("Delete");
	}	
}

// 2. Draw Points
setTool("multipoint");
waitForUser("Draw your points and click OK");
Roi.setName("Cell Locations");
roiManager("Add");

// 3a. Make Distance Map
getDimensions(x,y,c,z,t);
newImage(name+" - Distance to Top Map", "8-bit white", x, y, 1);
setForegroundColor(0, 0, 0);
roiManager("Select",0);
run("Draw", "slice");
setAutoThreshold("Default dark");
run("Convert to Mask");
run("Exact Euclidean Distance Transform (3D)");

// 4a. Measure Points
roiManager("Select", 2);
run("Measure");

// 3b. Make Distance Map
getDimensions(x,y,c,z,t);
newImage(name+" - Distance to Bottom Map", "8-bit white", x, y, 1);
setForegroundColor(0, 0, 0);
roiManager("Select",1);
run("Draw", "slice");
setAutoThreshold("Default dark");
run("Convert to Mask");
run("Exact Euclidean Distance Transform (3D)");

// 4b. Measure Points
roiManager("Select", 2);
run("Measure");

// Result appears in "Mean Column
// The order is such that the first "n" lines are the distances to the top edge
// The second "n" lines are the distances to the bottom edge.
