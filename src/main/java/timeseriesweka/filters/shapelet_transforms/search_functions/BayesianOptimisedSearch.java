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
import utilities.ClassifierTools;
import utilities.generic_storage.Pair;
import utilities.numericalmethods.NelderMead;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.RotationForest;
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

    @Override
    public void init(Instances data) {
        super.init(data);
    }

    @Override
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ShapeletSearch.ProcessCandidate checkCandidate) {

        evaluatedShapelets = new ArrayList<>();

        //do the random presamples.
        for (int i = 0; i < pre_samples; i++) {
            CandidateSearchData pair = GetRandomShapelet();
            evaluatePair(timeSeries, checkCandidate, pair);
        }

        current_gp = new GaussianProcesses();
        current_gp.setKernel(new RBFKernel()); //use RBF Kernel.

        for (int i = 0; i < num_iterations; i++) {

            try {
                Instances to_train = ConvertShapeletsToInstances(evaluatedShapelets);

                current_gp.buildClassifier(to_train);

                evaluatePair(timeSeries, checkCandidate, GetRandomShapeletFromGP(current_gp));
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

    
    public double GetIG(double[] params){
        
        int length = (int)params[1];
        //if we have invalid shapelet lengths of positions we want to fail the NelderMead.
        if (length < 3 || length > seriesLength)
            return 1E99;
        
        if(params[0] < 0 || params[0] >= seriesLength - length)
            return 1E99;
        
        try 
        {
            DenseInstance new_inst = new DenseInstance(3);
            new_inst.setValue(0, params[0]);
            new_inst.setValue(1, params[1]);
            new_inst.setValue(2, 0); //set it as 0, because we don't know it yet.

            return 1.0 - current_gp.classifyInstance(new_inst);
            
        } catch (Exception ex) {
            System.out.println("bad");
        }
        
        return 1E99;
    }
    
    public GaussianProcesses current_gp;
    
    public CandidateSearchData GetRandomShapeletFromGP(GaussianProcesses gp) throws Exception {
        
        NelderMead nm  = new NelderMead();
        
        evaluatedShapelets.sort(comparator);
        
        // from 0,3 -> best_current_shapelet, to max length shapelet.
        double[][] simplex = 
        {
            {0.0, 3.0},
            {evaluatedShapelets.get(0).startPos, evaluatedShapelets.get(0).length},
            {0.0, seriesLength}
        };
        
        nm.descend(this::GetIG, simplex);
        
        double[] params = nm.getResult();
        CandidateSearchData  bsf_pair = new CandidateSearchData((int)params[0], (int)params[1]);
        double bsf_ig= nm.getScore();

        System.out.println("predicted ig" + bsf_ig);
        System.out.println("bsf" + bsf_pair.getStartPosition() + bsf_pair.getLength());

        return bsf_pair;
    }

    public static void main(String[] args) throws Exception {

        String dir = "D:/Research TSC/Data/TSCProblems2018/";
        Instances[] data = DatasetLoading.sampleDataset(dir, "FordA", 1);

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
        
        RotationForest rotf = new RotationForest();
        rotf.buildClassifier(train);
        
        System.out.println(ClassifierTools.accuracy(test, rotf));

    }
}
