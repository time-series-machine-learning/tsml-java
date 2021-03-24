package tsml.classifiers.shapelet_based.quality;

import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceMV;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

public abstract class ShapeletQualityMV {

    protected double[][][] instancesArray;
    protected int[] classIndexes;
    protected String[] classNames;
    protected int[] classCounts;
    protected ShapeletDistanceMV distance;
    protected ArrayList<Attribute> atts;
    protected Instances instances;

    public ShapeletQualityMV(double[][][] instancesArray,
                              int[] classIndexes,
                              String[] classNames,
                              int[] classCounts,
                              ShapeletDistanceMV distance){
        this.instancesArray = instancesArray;
        this.classIndexes = classIndexes;
        this.classNames = classNames;
        this.classCounts = classCounts;
        this.distance = distance;
        this.atts = new ArrayList<Attribute>(2);
        this.atts.add(new Attribute("distance"));
        this.atts.add(new Attribute("class", Arrays.asList(classNames)));

    }

    public ShapeletQualityMV() {

    }

    public abstract double calculate(ShapeletMV candidate);


}
