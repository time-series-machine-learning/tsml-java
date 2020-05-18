/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers.legacy.COTE;

import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.ElasticEnsemble;
import java.util.ArrayList;
import java.util.Random;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.filters.shapelet_filters.ShapeletFilter;
import tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities;
import utilities.ClassifierTools;
import machine_learning.classifiers.ensembles.CAWPE;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import tsml.transformers.ACF;
import tsml.transformers.PowerSpectrum;
import weka.core.Randomizable;
import weka.core.TechnicalInformationHandler;
/**
 * NOTE: consider this code legacy. There is no reason to use FlatCote over HiveCote. 
 * Also note that file writing/reading from file is not currently supported (will be added soon)
 
 @article{bagnall15cote,
  title={Time-Series Classification with {COTE}: The Collective of Transformation-Based Ensembles},
  author={A. Bagnall and J. Lines and J. Hills and A. Bostrom},
  journal={{IEEE} Transactions on Knowledge and Data Engineering},
  volume={27},
  issue={9},
  pages={2522--2535},
  year={2015}
}

 
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class FlatCote extends EnhancedAbstractClassifier implements TechnicalInformationHandler{

    public FlatCote() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
    }

      
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "A. Bagnall and J. Lines and J. Hills and A. Bostrom");
        result.setValue(TechnicalInformation.Field.TITLE, "Time-Series Classification with COTE: The Collective of Transformation-Based Ensembles");
        result.setValue(TechnicalInformation.Field.JOURNAL, "IEEE Transactions on Knowledge and Data Engineering");
        result.setValue(TechnicalInformation.Field.VOLUME, "27");
        result.setValue(TechnicalInformation.Field.NUMBER, "9");
        
        result.setValue(TechnicalInformation.Field.PAGES, "2522-2535");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        return result;
    }    
    // Flat-COTE includes 35 constituent classifiers:
    //  -   11 from the Elastic Ensemble
    //  -   8 from the Shapelet Transform Ensemble
    //  -   8 from CAWPE (ACF transformed)
    //  -   8 from CAWPE (PS transformed)
    private Instances train;
    
    
    private ElasticEnsemble ee;
    private CAWPE st;
    private CAWPE acf;
    private CAWPE ps;
    private int numClassifiers=0;
//    private ShapeletTransform shapeletTransform;
    private double[][] cvAccs;
    private double cvSum;
    
    private double[] weightByClass;


    
    @Override
    public void buildClassifier(Instances train) throws Exception{

        long t1=System.nanoTime();
        this.train = train;
        
        ee = new ElasticEnsemble();
        ShapeletTransformClassifier stc = new ShapeletTransformClassifier();
        stc.setHourLimit(24);
        stc.setClassifier(new CAWPE());
//Redo for STC
        //ShapeletTransform shapeletTransform = ShapeletTransformFactory.createTransform(train);
        ShapeletFilter shapeletFilter = ShapeletTransformTimingUtilities.createTransformWithTimeLimit(train, 24); // now defaults to max of 24 hours
        shapeletFilter.supressOutput();
        st = new CAWPE();
        //st.setTransform(shapeletFilter); //TODO: Update Shapelets so i can update CAWPE
        st.setupOriginalHESCASettings();
        acf = new CAWPE();
        acf.setupOriginalHESCASettings();
        acf.setTransform(new ACF());
        ps = new CAWPE();
        ps.setupOriginalHESCASettings();
        ps.setTransform(new PowerSpectrum());

        if(seedClassifier){
            if(ee instanceof Randomizable)
                ((Randomizable)ee).setSeed(seed);
            if(st instanceof Randomizable)
                ((Randomizable)st).setSeed(seed);
            if(acf instanceof Randomizable)
                ((Randomizable)st).setSeed(seed);
            if(acf instanceof Randomizable)
                ((Randomizable)st).setSeed(seed);
            
        }
//        st.setDebugPrinting(true);
        ee.buildClassifier(train);
        acf.buildClassifier(train);
        ps.buildClassifier(train);
        st.buildClassifier(train);
        
        cvAccs = new double[4][];
        cvAccs[0] = ee.getCVAccs();
        cvAccs[1] = st.getIndividualAccEstimates();
        cvAccs[2] = acf.getIndividualAccEstimates();
        cvAccs[3] = ps.getIndividualAccEstimates();
        
        cvSum = 0;
        for(int e = 0; e < cvAccs.length;e++){
            for(int c = 0; c < cvAccs[e].length; c++){
                cvSum+=cvAccs[e][c];
            }
        }
        long t2=System.nanoTime();
        trainResults.setBuildTime(t2-t1);
        for(int i=0;i<cvAccs.length;i++)
            numClassifiers+=cvAccs[i].length;
    }
    
    @Override
    public double[] distributionForInstance(Instance test) throws Exception{
        weightByClass = null;
        classifyInstance(test);
        double[] dists = new double[weightByClass.length];
        for(int c = 0; c < weightByClass.length; c++){
            dists[c] = weightByClass[c]/this.cvSum;
        }
        return dists;
    }
    
    @Override
    public double classifyInstance(Instance test) throws Exception{
        
        double[][] preds = new double[4][];
        
        preds[0] = this.ee.classifyInstanceByConstituents(test);
        preds[1] = this.st.classifyInstanceByConstituents(test);
        preds[2] = this.acf.classifyInstanceByConstituents(test);
        preds[3] = this.ps.classifyInstanceByConstituents(test);
        
        weightByClass = new double[train.numClasses()];
        ArrayList<Double> bsfClassVals = new ArrayList<>();
        double bsfWeight = -1;
        
        for(int e = 0; e < preds.length; e++){
            for(int c = 0; c < preds[e].length; c++){
                weightByClass[(int)preds[e][c]]+=cvAccs[e][c];
//                System.out.print(preds[e][c]+",");
                if(weightByClass[(int)preds[e][c]] > bsfWeight){
                    bsfWeight = weightByClass[(int)preds[e][c]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(preds[e][c]);
                }else if(weightByClass[(int)preds[e][c]] > bsfWeight){
                    bsfClassVals.add(preds[e][c]);
                }
            }
        }
        
        if(bsfClassVals.size()>1){
            return bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
        }        
        return bsfClassVals.get(0);
    }
    @Override
    public String getParameters() {
        String str=super.getParameters();
        str+=",NumClassifiers,"+numClassifiers+",EE,"+cvAccs[0].length+",ACF_"+acf.getEnsembleName()+","+cvAccs[1].length+",PS_"+ps.getEnsembleName()+","+cvAccs[2].length+",ST_"+st.getEnsembleName()+","+cvAccs[3].length+",CVAccs,";
        for(int i=0;i<cvAccs.length;i++)
            for(int j=0;j<cvAccs[i].length;j++)
                str+=cvAccs[i][j]+",";
        
        return str;
    }


    
    public static void main(String[] args) throws Exception{
        
//        System.out.println(ClassifierTools.testUtils_getIPDAcc(new FlatCote()));
        
        FlatCote fc = new FlatCote();
        String datasetName = "Chinatown";
        
        Instances train = DatasetLoading.loadDataNullable("Z:/ArchiveData/Univariate_arff/"+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = DatasetLoading.loadDataNullable("Z:/ArchiveData/Univariate_arff/"+datasetName+"/"+datasetName+"_TEST");
        System.out.println("Example usage of HiveCote: this is the code used in the paper");
        System.out.println(fc.getTechnicalInformation().toString());
        System.out.println("Evaluated on "+datasetName);
 
        fc.buildClassifier(train);
        System.out.println("Build is complete");
        System.out.println("Flat Cote parameters :"+fc.getParameters());
        
        double a=ClassifierTools.accuracy(test, fc);
        System.out.println("Test acc for "+datasetName+" = "+a);
        
                
    }
    
}
