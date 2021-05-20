package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;

public class RandomFilterByClass extends RandomFilter {


    @Override
    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {
        long start = System.nanoTime();
        int numClasses = instances.getClassCounts().length;
        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances,
                params.distance.createShapeletDistance());


        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        TimeSeriesInstances[] instancesArray = new TimeSeriesInstances[numClasses];

        for (int i=0;i<numClasses;i++){
            instancesArray[i] = new TimeSeriesInstances(instances.getClassLabels());

        }
        for (int j=0;j<instances.numInstances();j++){
            instancesArray[instances.get(j).getLabelIndex()].add(instances.get(j));
        }

        for (int i=0;i<numClasses;i++){

              System.out.println("Shapelets for class " + i + "/" + numClasses);
            ArrayList<ShapeletMV> classShapelets = randomShapelets(params,instancesArray[i],start, type, quality, params.maxIterations/numClasses, params.k/numClasses);
        //    System.out.println(classShapelets);
            shapelets.addAll(classShapelets);
        }


        return shapelets;
    }



}
