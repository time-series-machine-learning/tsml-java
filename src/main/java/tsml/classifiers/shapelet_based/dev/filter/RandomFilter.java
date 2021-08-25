package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RandomFilter extends ShapeletFilterMV {

    protected int iteration;
    protected double averageQuality;
    protected MSTC.ShapeletParams params;
    protected StopCriteria stopCriteria;
    protected ArrayList<ShapeletMV> shapelets;

    @Override
    public List<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {

        this.start = System.nanoTime();
        this.params = params;

        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances);

        return findShapelets(params,quality,instances);


    }

    @Override
    public List<ShapeletMV> findShapelets(MSTC.ShapeletParams params, ShapeletQualityFunction quality,
                                               TimeSeriesInstances instances) {

        ShapeletFunctions fun = params.type.createShapeletType();

        this.shapelets = new ArrayList<ShapeletMV>();

        int[] classesArray  = instances.getClassIndexes();

        this.stopCriteria = new Combined();
        this.params = params;
        this.start = System.nanoTime();
        iteration = 0;
        while (true){ // Iterate

        //    if (iteration % (params.maxIterations/10) == 0){
         //       System.out.println("Iteration: "+ iteration);
        //    }

            //Get random shapelet
            int shapeletSize = params.min + (int)(MSTC.RAND.nextInt(params.max-params.min));
            int instanceIndex =  MSTC.RAND.nextInt(instances.numInstances());
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
                System.out.println("It.:" + iteration +  " Avg. Quality: " + averageQuality  +
                        " Num shapelets: " + shapelets.size());
                if (this.stopCriteria.stop()){
                    return shapelets;
                }

            }
            iteration++;
        }
    }

    protected static ArrayList<ShapeletMV> removeSelfSimilar(ShapeletFunctions fun, ArrayList<ShapeletMV> shapelets) {
        // return a new pruned array list - more efficient than removing
        // self-similar entries on the fly and constantly reindexing
        ArrayList<ShapeletMV> outputShapelets = new ArrayList<>();
        int size = shapelets.size();

        boolean[] selfSimilar = new boolean[size];

        for (int i = 0; i < size; i++) {
            if (selfSimilar[i]) {
                continue;
            }

            outputShapelets.add(shapelets.get(i));

            for (int j = i + 1; j < size; j++) {
                // no point recalc'ing if already self similar to something
                if ((!selfSimilar[j]) && fun.selfSimilarity(shapelets.get(i), shapelets.get(j))) {
                    selfSimilar[j] = true;
                }
            }
        }
        return outputShapelets;
    }


    interface StopCriteria{
        public boolean stop();
    }

    class Iterations implements StopCriteria{
        public boolean stop(){
            return iteration>=params.maxIterations;
        }
    }

    class Contracted implements StopCriteria{
        @Override
        public boolean stop() {
            return withinTrainContract(start);
        }
    }

    class BestFound implements StopCriteria{
        @Override
        public boolean stop() {
            return averageQuality>0.999;
        }
    }

    class NoImprovement implements StopCriteria{

        double prevImprovement = 1;
        @Override
        public boolean stop() {
            if (shapelets.size() >= params.k){
                if (prevImprovement==averageQuality) return true;
                else{
                    prevImprovement = averageQuality;
                    return  false;
                }

            }else{
                return false;
            }
        }
    }

    class Combined implements StopCriteria{
        @Override
        public boolean stop() {
            return shapelets.size()>0 && (iteration>=params.maxIterations ||
                    withinTrainContract(start));
        }
    }




}
