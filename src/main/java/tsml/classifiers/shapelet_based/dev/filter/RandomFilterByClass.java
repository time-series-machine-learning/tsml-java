package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RandomFilterByClass extends RandomFilter {


    @Override
    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {
        this.start = System.nanoTime();

        int numClasses = instances.getClassCounts().length;
        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances);


        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        ArrayList[] instancesOptions = new ArrayList[numClasses];

        for (int i=0;i<numClasses;i++){
            instancesOptions[i] = new ArrayList<Integer>();

        }
        for (int j=0;j<instances.numInstances();j++){
            instancesOptions[instances.get(j).getLabelIndex()].add(j);
        }

        RandomFilter filter = new RandomFilter();
        filter.setHourLimit(params.contractTimeHours / numClasses);
        MSTC.ShapeletParams classParams = new MSTC.ShapeletParams(params);
        classParams.k = params.k / numClasses;
        classParams.maxIterations = params.maxIterations / numClasses;
        for (int i=0;i<numClasses;i++){
            System.out.println("Shapelets for class " + i + "/" + numClasses);
            List<ShapeletMV> classShapelets = findShapeletsMulti(instancesOptions[i], classParams, quality, instances);
            shapelets.addAll(classShapelets);
        }
        return shapelets;
    }


    public List<ShapeletMV> findShapeletsMulti(ArrayList<Integer> instancesOption,
                                               MSTC.ShapeletParams classParams,
                                               ShapeletQualityFunction quality,
                                               TimeSeriesInstances instances)
    {

        ShapeletFunctions fun = classParams.type.createShapeletType();

        this.shapelets = new ArrayList<ShapeletMV>();

        int[] classesArray  = instances.getClassIndexes();

        Combined stopCriteria = new Combined();
        this.params = classParams;
        this.start = System.nanoTime();
        iteration = 0;
        while (true){ // Iterate

            if (iteration % (params.maxIterations/10) == 0){
                System.out.println("Iteration: "+ iteration);
            }

            //Get random shapelet
            int shapeletSize = params.min + (int)(MSTC.RAND.nextInt(params.max-params.min));
            shapeletSize = Math.min(shapeletSize,3000);

            int instanceIndex =  instancesOption.get(MSTC.RAND.nextInt(instancesOption.size()));

            ShapeletMV candidate = fun.getRandomShapelet(
                    shapeletSize,
                    instanceIndex,
                    classesArray[instanceIndex],
                    instances.get(instanceIndex));
            double q = quality.calculate (fun, candidate);
            //   System.out.println(q);
            //    if (!isSimilar(shapelets, candidate,type,params.minDist)){

            if (q>0){
                candidate.setQuality(q);
                shapelets.add(candidate);

            }

            //   }

            if (iteration % 1000 == 0){
                Collections.sort(shapelets);
                if ( shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
                this.shapelets = removeSelfSimilar(fun,this.shapelets);

                averageQuality = shapelets.stream().mapToDouble(ShapeletMV::getQuality).average().orElse(0);
                System.out.println(averageQuality  + " " + shapelets.size());
                if (stopCriteria.stop()){
                    return shapelets;
                }

            }
            iteration++;
        }
    }

}
