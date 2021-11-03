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
    protected ArrayList<ShapeletMV> shapelets;

    protected IterationsStopCriteria iterationsStopCriteria;
    protected ContractedStopCriteria contractedStopCriteria;
    protected QualityStopCriteria qualityStopCriteria;

    protected int numShapeletsEvaluated;
    protected String endCriteria;

    @Override
    public List<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {

        this.start = System.nanoTime();
        this.params = params;
        this.iterationsStopCriteria = new IterationsStopCriteria();
        this.contractedStopCriteria = new ContractedStopCriteria();
        this.qualityStopCriteria = new QualityStopCriteria();

        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances);

        return findShapelets(params,quality,instances);


    }

    @Override
    public List<ShapeletMV> findShapelets(MSTC.ShapeletParams params, ShapeletQualityFunction quality,
                                               TimeSeriesInstances instances) {

        ShapeletFunctions fun = params.type.createShapeletType();

        this.shapelets = new ArrayList<ShapeletMV>();

        int[] classesArray  = instances.getClassIndexes();

        this.params = params;
        this.start = System.nanoTime();
        this.iteration = 0;
        this.numShapeletsEvaluated = 0;
        while (true){ // Iterate

            //Get random shapelet
            int shapeletSize = params.min + (int)(MSTC.RAND.nextInt(params.max-params.min));
            int instanceIndex =  MSTC.RAND.nextInt(instances.numInstances());
            ShapeletMV candidate = fun.getRandomShapelet(
                    shapeletSize,
                    instanceIndex,
                    classesArray[instanceIndex],
                    instances.get(instanceIndex));
            double q = quality.calculate (fun, candidate);

            this.numShapeletsEvaluated++;

            if (q>0 || params.allowZeroQuality){
                candidate.setQuality(q);
                shapelets.add(candidate);

            }

            if (this.iteration % 1000 == 0){
                reorderShapelets(fun);
            }
            if (this.iterationsStopCriteria.stop()){
                reorderShapelets(fun);
                this.endCriteria = "MAX_ITER";
                return shapelets;
            }
            if (this.contractedStopCriteria.stop()){
                reorderShapelets(fun);
                this.endCriteria = "CONTRACT_TIME";
                return shapelets;
            }

         /*   if (this.qualityStopCriteria.stop()){
                reorderShapelets(fun);
                System.out.println("Quality reached");
                System.out.println("num shapelets evaluated: " + numShapeletsEvaluated);
                return shapelets;
            }*/

            this.iteration++;
        }
    }

    protected void reorderShapelets(ShapeletFunctions fun){
        Collections.sort(this.shapelets);
        if ( this.shapelets.size()>this.params.k)
            this.shapelets.subList(this.params.k,this.shapelets.size()).clear();
        if ( this.params.removeSelfSimilar )
            this.shapelets = removeSelfSimilar(fun,this.shapelets);

        this.averageQuality = this.shapelets.stream().mapToDouble(ShapeletMV::getQuality).average().orElse(0);

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

    @Override
    public String getParameters(){
        return "End criteria, " + endCriteria + ", shapelets evaluated," + numShapeletsEvaluated + ", avg quality," + averageQuality;
    }


    interface StopCriteria{
        public boolean stop();
    }

    class IterationsStopCriteria implements StopCriteria{
        public boolean stop(){
            return iteration>=params.maxIterations;
        }
    }

    class ContractedStopCriteria implements StopCriteria{
        @Override
        public boolean stop() {
            return withinTrainContract(start);
        }
    }

    class QualityStopCriteria implements StopCriteria{
        @Override
        public boolean stop() {
            return shapelets.size() >= params.k*0.9 && averageQuality>0.95;
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
            return  (iteration>=params.maxIterations ||
                    withinTrainContract(start));
        }
    }




}
