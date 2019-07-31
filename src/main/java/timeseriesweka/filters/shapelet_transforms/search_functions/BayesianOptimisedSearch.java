/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.search_functions;

import experiments.data.DatasetLoading;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransform;
import utilities.generic_storage.Pair;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author a.bostrom1
 */
public class BayesianOptimisedSearch extends ImpRandomSearch {

    public ArrayList<Shapelet> evaluatedShapelets; //not the right type yet.

    public int pre_samples = 100;
    public int num_iterations = 100;

    public BayesianOptimisedSearch(ShapeletSearchOptions ops) {
        super(ops);
    }

    ArrayList<CandidateSearchData> shapelet_params = new ArrayList();

    @Override
    public void init(Instances data) {
        super.init(data);

        //rather than the whole param grid, could try doing some kind of gradient descent.
        //generate the param grid.
        for (int length = minShapeletLength; length <= maxShapeletLength; length += lengthIncrement) {
            //for all possible starting positions of that length. -1 to remove classValue but would be +1 (m-l+1) so cancel.
            for (int start = 0; start < seriesLength - length; start += positionIncrement) {
                shapelet_params.add(new CandidateSearchData(start,length));
            }
        }
    }

    @Override
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ShapeletSearch.ProcessCandidate checkCandidate) {

        evaluatedShapelets = new ArrayList<>();

        //do the random presamples.
        for (int i = 0; i < pre_samples; i++) {
            CandidateSearchData pair = GetRandomShapelet();
            evaluatePair(timeSeries, checkCandidate, pair);
        }

        GaussianProcesses gp = new GaussianProcesses();
        gp.setKernel(new RBFKernel()); //use RBF Kernel.

        for (int i = 0; i < num_iterations; i++) {

            try {
                Instances to_train = ConvertShapeletsToInstances(evaluatedShapelets);

                gp.buildClassifier(to_train);

                evaluatePair(timeSeries, checkCandidate, GetRandomShapeletFromGP(gp));
            } catch (Exception ex) {
                Logger.getLogger(BayesianOptimisedSearch.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return evaluatedShapelets;
    }

    public Instances ConvertShapeletsToInstances(ArrayList<Shapelet> shapelets) {
        FastVector atts = new FastVector();
        atts.addElement(new Attribute("Length"));
        atts.addElement(new Attribute("StartPosition"));
        atts.addElement(new Attribute("QualityValue"));

        //same number of xInstances 
        Instances result = new Instances("shapelets", atts, shapelets.size());

        for (int i = 0; i < shapelets.size(); i++) {
            result.add(new DenseInstance(3));
            result.instance(i).setValue(0, shapelets.get(i).length);
            result.instance(i).setValue(1, shapelets.get(i).startPos);
            result.instance(i).setValue(2, shapelets.get(i).qualityValue);
        }

        result.setClassIndex(2);

        return result;
    }

    public Instance ConvertPairToInstance(CandidateSearchData pair) {
        DenseInstance new_inst = new DenseInstance(3);
        new_inst.setValue(0, pair.getStartPosition());
        new_inst.setValue(1, pair.getLength());
        new_inst.setValue(2, 0); //set it as 0, because we don't know it yet.

        return new_inst;
    }

    public Shapelet evaluatePair(Instance timeSeries, ShapeletSearch.ProcessCandidate checkCandidate, CandidateSearchData pair) {
        Shapelet shape = checkCandidate.process(timeSeries, pair.getStartPosition(), pair.getLength());
        
        System.out.println("quality value: "+ shape.qualityValue);
        evaluatedShapelets.add(shape);

        return shape;
    }

    public CandidateSearchData GetRandomShapelet() {
        int numLengths = maxShapeletLength - minShapeletLength; //want max value to be inclusive.
        int length = random.nextInt(numLengths) + minShapeletLength; //offset the index by the min value.
        int position = random.nextInt(seriesLength - length); // can only have valid start positions based on the length. (numAtts-1)-l+1
        //find the shapelets for that series.

        //add the random shapelet to the length
        return new CandidateSearchData(position,length);
    }

    public CandidateSearchData GetRandomShapeletFromGP(GaussianProcesses gp) throws Exception {

        //given a gp. loop through the param grid, and find the highest quality value, then return that.
        CandidateSearchData bsf_pair = shapelet_params.get(0);
        double bsf_ig = gp.classifyInstance(ConvertPairToInstance(bsf_pair));
        int bsf_index = 0;

        
        /*PriorityQueue<Pair<Integer, Integer>> eval_list = new PriorityQueue();
        
        random.nextInt(shapelet_params.size());*/
        //could use some kind of random selection with gradient descent.
        for(int i=1; i<shapelet_params.size(); i++){
            CandidateSearchData pair = shapelet_params.get(i);
            double ig = gp.classifyInstance(ConvertPairToInstance(pair));

            if (ig > bsf_ig) {
                bsf_ig = ig;
                bsf_pair = pair;
                bsf_index = i;
            }
        }

        System.out.println("predicted ig" + bsf_ig);
        System.out.println("bsf" + bsf_pair.getStartPosition() + bsf_pair.getLength());

        return bsf_pair;
    }

    public static void main(String[] args) throws Exception {

        String dir = "D:/Research TSC/Data/TSCProblems2018/";
        Instances[] data = DatasetLoading.sampleDataset(dir, "ItalyPowerDemand", 1);

        Instances train = data[0];
        Instances test = data[1];

        int m = train.numAttributes() - 1;

        ShapeletSearchOptions sops = new ShapeletSearchOptions.Builder()
                .setSearchType(ShapeletSearch.SearchType.BO_SEARCH)
                .setMin(3).setMax(m)
                .setSeed(1)
                .setNumShapelets(100)
                .build();

        ShapeletTransform st = new ShapeletTransform();
        st.setSearchFunction(new ShapeletSearchFactory(sops).getShapeletSearch());

        st.process(train);
        st.process(test);

    }
}
