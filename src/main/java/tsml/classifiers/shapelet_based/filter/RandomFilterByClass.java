package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.type.ShapeletFunctions;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class RandomFilterByClass extends RandomFilter {


    @Override
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params,
                                               TimeSeriesInstances instances) {
        long start = System.nanoTime();
        int numClasses = instances.getClassCounts().length;
        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances,
                params.distance.createShapeletDistance());


        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        for (int i=0;i<numClasses;i++){
            System.out.println("Class "  + i);
            final int classindex = i;
            List<TimeSeriesInstance> ci = instances.stream().filter(instance -> instance.getTargetValue()==classindex).collect(Collectors.toList());
            TimeSeriesInstances classInstances = new TimeSeriesInstances(ci, instances.getClassLabels());
            ArrayList<ShapeletMV> classShapelets = randomShapelets(params,classInstances,start, type, quality, params.maxIterations/numClasses, params.k/numClasses);
            shapelets.addAll(classShapelets);
        }


        return shapelets;
    }



}
