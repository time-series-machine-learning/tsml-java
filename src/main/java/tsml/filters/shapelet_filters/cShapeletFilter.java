package tsml.filters.shapelet_filters;
/**
 * This is a first go at introducing a simple time contract that uses time rather than an estimate of number of shapelets
 * and does it internally rather than externally.
 *
 * This is a hacked version, and the whole structure could be tidied up. One problem is that the factory decides on
 * BalancedClassShapeletTransform and ShapeletTransform based on the number of classes. This means if cShapeletTransform
 * extends BalancedClassShapeletTransform, it needs to internally revert to ShapeletTransform method. Essentially breaking
 * encapsulation by modelling the behaviour super.super.findBestKShapeletsCache (whichof course is not allowed).
 * It also makes configuring the builder/factory model harder. cShapeletTransform needs useBalanced to be set.
 *
 * this can be made better by absorbing the BalancedClassShapeletTransform into ShapeletTransform and just switching there
 * instead of here.
 *
 * in terms of the contract, there is also the issue of the time taken to perform the transform. This can be quite long for
 * big data
 */

import tsml.transformers.shapelet_tools.Shapelet;
import weka.core.Instances;

import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.TreeMap;

public class cShapeletFilter extends BalancedClassShapeletFilter {
    private int numSeriesToUse=0;
    private long contractTime=0; //nano seconds time. If set to zero everything reverts to BalancedClassShapeletTransform
    public void setContractTime(long c){
        contractTime=c;

    }
    @Override
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances data) {
        if (useBalancedClasses)
            return findBestKShapeletsCacheBalanced(data);
        else
            return findBestKShapeletsCacheOriginal(data);
    }

    /**
     *
     * @param data
     * @return
     */
    private ArrayList<Shapelet> findBestKShapeletsCacheBalanced(Instances data) {
        if(contractTime==0)
            return super.findBestKShapeletsCache(data);
        long startTime=System.nanoTime();
        long usedTime=0;
        int numSeriesToUse = data.numInstances(); //This can be used to reduce the number of series in favour of more
        System.out.println(" Set up in contract balanced in cST");
        System.out.println("\t\t\t numShapelets "+numShapelets);
        System.out.println("\t\t\t Contract (secs) = "+contractTime/1000000000.0);
        System.out.println("Search function  "+searchFunction.getSearchType());
        System.out.println("Shapelets per series  "+getNumShapeletsPerSeries());


        ArrayList<Shapelet> seriesShapelets; // temp store of all shapelets for each time series
        //construct a map for our K-shapelets lists, on for each classVal.

        if(kShapeletsMap == null){
            kShapeletsMap = new TreeMap();
            for (int i=0; i < data.numClasses(); i++){
                kShapeletsMap.put((double)i, new ArrayList<>());
            }
        }

        //found out how many we want in each sub list.
        int proportion = numShapelets/kShapeletsMap.keySet().size();

        //for all time series
        outputPrint("Processing data for numShapelets "+numShapelets+ " with proportion per class = "+proportion);
        outputPrint("in contract balanced: Contract (secs)"+contractTime/1000000000.0);

        //continue processing series until we run out of time
        while(casesSoFar < numSeriesToUse && usedTime<contractTime)
        {
            System.out.println(casesSoFar +" Cumulative time (secs) = "+usedTime/1000000000.0);
            //get the Shapelets list based on the classValue of our current time series.
            kShapelets = kShapeletsMap.get(data.get(casesSoFar).classValue());
            //we only want to pass in the worstKShapelet if we've found K shapelets. but we only care about this class values worst one.
            //this is due to the way we represent each classes shapelets in the map.
            worstShapelet = kShapelets.size() == proportion ? kShapelets.get(kShapelets.size()-1) : null;

            //set the series we're working with.
            subseqDistance.setSeries(casesSoFar);
            //set the class value of the series we're working with.
            classValue.setShapeletValue(data.get(casesSoFar));
            seriesShapelets = searchFunction.searchForShapeletsInSeries(data.get(casesSoFar), this::checkCandidate);

//Here we can tweak the the number of shapelets to do per series, although it would be much easier with time.
            numShapeletsEvaluated+=seriesShapelets.size();
//            outputPrint("BalancedClassST: data : " + casesSoFar+" has "+seriesShapelets.size()+" candidates"+ " cumulative early abandons "+numEarlyAbandons);
            if(seriesShapelets != null){
                Collections.sort(seriesShapelets, shapeletComparator);
                if(isRemoveSelfSimilar())
                    seriesShapelets = removeSelfSimilar(seriesShapelets);

                kShapelets = combine(proportion, kShapelets, seriesShapelets);
            }

            //re-update the list because it's changed now.
            kShapeletsMap.put(data.get(casesSoFar).classValue(), kShapelets);

            casesSoFar++;

            createSerialFile();
            usedTime=System.nanoTime()-startTime;
            //Logic is we have underestimated the contract so can run back through. If we over estimate it we will just stop.
            if(casesSoFar==numSeriesToUse-1 && !searchFunction.getSearchType().equals("FULL")) ///HORRIBLE!
                casesSoFar=0;

        }

        kShapelets = buildKShapeletsFromMap(kShapeletsMap);

        this.numShapelets = kShapelets.size();


        if (recordShapelets)
            recordShapelets(kShapelets, this.ouputFileLocation);
        if (!supressOutput)
            writeShapelets(kShapelets, new OutputStreamWriter(System.out));

        return kShapelets;
    }

    public ArrayList<Shapelet> findBestKShapeletsCacheOriginal(Instances data) {
        long time=contractTime;
        if(time==0)
            time= Long.MAX_VALUE; //If no contract, keep going until all series looked at
        long startTime=System.nanoTime();
        long usedTime=0;
        int numSeriesToUse = data.numInstances(); //This can be used to reduce the number of series in favour of more

        ArrayList<Shapelet> seriesShapelets;                                    // temp store of all shapelets for each time series
        // temp store of all shapelets for each time series
        //for all time series
        System.out.println(" Set up in contract original cache in cST");
        System.out.println("\t\t\t numShapelets "+numShapelets);
        System.out.println("\t\t\t Contract (secs) = "+contractTime/1000000000.0);
        System.out.println("Search function  "+searchFunction.getSearchType());
        System.out.println("Shapelets per series  "+getNumShapeletsPerSeries());
//        System.out.println("\t\t\t number per series = "+contractTime/1000000000.0);
//        System.exit(0);
        int dataSize = data.numInstances();
        //for all possible time series.

        for(; casesSoFar < numSeriesToUse && usedTime<time; casesSoFar++) {
            System.out.println(casesSoFar +" Cumulative time (secs) = "+usedTime/1000000000.0);
            //set the worst Shapelet so far, as long as the shapelet set is full.
            worstShapelet = kShapelets.size() == numShapelets ? kShapelets.get(numShapelets - 1) : null;

            //set the series we're working with.
            subseqDistance.setSeries(casesSoFar);
            //set the class value of the series we're working with.
            classValue.setShapeletValue(data.get(casesSoFar));

            seriesShapelets = searchFunction.searchForShapeletsInSeries(data.get(casesSoFar), this::checkCandidate);
            numShapeletsEvaluated+=seriesShapelets.size();
            outputPrint("data : " + casesSoFar+" has "+seriesShapelets.size()+" candidates"+ " cumulative early abandons "+numEarlyAbandons+" worst so far ="+worstShapelet);
            if(seriesShapelets != null){
                Collections.sort(seriesShapelets, shapeletComparator);

                if(isRemoveSelfSimilar())
                    seriesShapelets = removeSelfSimilar(seriesShapelets);
                kShapelets = combine(numShapelets, kShapelets, seriesShapelets);
            }

            createSerialFile();
            usedTime=System.nanoTime()-startTime;
            //Logic is we have underestimated the contract so can run back through. If we over estimate it we will just stop.
            if(casesSoFar==numSeriesToUse-1 && !searchFunction.getSearchType().equals("FULL")) ///HORRIBLE!
                casesSoFar=0;
        }
        this.numShapelets = kShapelets.size();

        if (recordShapelets)
            recordShapelets(kShapelets, this.ouputFileLocation);
        if (!supressOutput)
            writeShapelets(kShapelets, new OutputStreamWriter(System.out));
        System.out.println("Time used in find k shapelets = "+usedTime/1000000000.0+" leaving the method");

        return kShapelets;
    }


}
