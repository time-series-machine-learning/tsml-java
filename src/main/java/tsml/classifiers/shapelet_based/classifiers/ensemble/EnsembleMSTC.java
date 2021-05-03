package tsml.classifiers.shapelet_based.classifiers.ensemble;

import tsml.classifiers.TSClassifier;
import tsml.classifiers.shapelet_based.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstances;
import weka.classifiers.Classifier;

public class EnsembleMSTC extends AbstractEnsembleTS  {




    public EnsembleMSTC() {
        super();
    }


    @Override //Abstract Ensemble
    public final void setupDefaultEnsembleSettings(TimeSeriesInstances data) {
        this.ensembleName = "CAWPE";

        this.weightingScheme = new EqualWeightingTS();
        this.votingScheme = new MajorityConfidenceTS();

        this.numEnsembles = 100;
        TSClassifier[] classifiers = new MSTC[numEnsembles];
        MSTC.ShapeletParams params = new MSTC.ShapeletParams(500,
                5,data.getMinLength()-1,
                100000,0.01,
                MSTC.ShapeletFilters.RANDOM, MSTC.ShapeletQualities.GAIN_RATIO,
                MSTC.ShapeletDistances.EUCLIDEAN,
                MSTC.ShapeletFactories.DEPENDANT,
                MSTC.AuxClassifiers.LINEAR);
        for (int i=0;i<numEnsembles;i++){
            classifiers[i] = new MSTC(params);
        }


        setClassifiers(classifiers);
    }


    @Override
    public Classifier getClassifier() {
        return null;
    }

    @Override
    public TimeSeriesInstances getTSTrainData() {
        return null;
    }

    @Override
    public void setTSTrainData(TimeSeriesInstances train) {

    }


}
