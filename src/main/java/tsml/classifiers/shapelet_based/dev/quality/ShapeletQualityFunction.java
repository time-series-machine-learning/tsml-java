package tsml.classifiers.shapelet_based.dev.quality;

import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

public abstract class ShapeletQualityFunction {

    protected TimeSeriesInstances trainInstances;
    protected int[] classIndexes;
    protected String[] classNames;
    protected int[] classCounts;
    protected ShapeletDistanceFunction distance;
    protected ArrayList<Attribute> atts;
    protected Instances instances;

    public ShapeletQualityFunction(TimeSeriesInstances instances,
                                   ShapeletDistanceFunction distance){
        this.trainInstances = instances;
        this.classIndexes = instances.getClassIndexes();
        this.classNames = instances.getClassLabels();
        this.classCounts = instances.getClassCounts();
        this.distance = distance;
        this.atts = new ArrayList<Attribute>(2);
        this.atts.add(new Attribute("distance"));
        this.atts.add(new Attribute("class", Arrays.asList(classNames)));

    }

    public ShapeletQualityFunction() {

    }

    public abstract double calculate(ShapeletMV candidate);



}
