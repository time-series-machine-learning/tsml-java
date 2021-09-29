package tsml.classifiers.shapelet_based.dev.classifiers.ensemble;

import tsml.classifiers.TSClassifier;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstances;
import weka.classifiers.Classifier;

import java.util.Random;

public class CombinedEnsembleMSTC extends AbstractEnsembleTS {

    MSTC.ShapeletParams params;

    public CombinedEnsembleMSTC(MSTC.ShapeletParams params) {
        super();
        this.params = params;
    }

    @Override //Abstract Ensemble
    public final void setupDefaultEnsembleSettings(TimeSeriesInstances data) {
        this.ensembleName = "5BIN-MSTC";

        this.weightingScheme = new EqualWeightingTS();
        this.votingScheme = new MajorityConfidenceTS();



        Random r = new Random();

       // MSTC.ShapeletQualities[] QUALITIES = MSTC.ShapeletQualities.values();
        MSTC.ShapeletFactories[] TYPES =  MSTC.ShapeletFactories.values();
        MSTC.ShapeletQualities[] QUALITIES =  {
                MSTC.ShapeletQualities.GAIN_BINARY,
                MSTC.ShapeletQualities.CHI_BINARY,
                MSTC.ShapeletQualities.CORR_BINARY,
                MSTC.ShapeletQualities.FSTAT_BINARY,
                MSTC.ShapeletQualities.ONE_R_BINARY, };
        this.numEnsembles = QUALITIES.length;
        TSClassifier[] classifiers = new MSTC[numEnsembles];

        for (int i=0;i<numEnsembles;i++){

            MSTC.ShapeletParams params = new MSTC.ShapeletParams(this.params);
            params.quality = QUALITIES[i];
            params.contractTimeHours = 2;
            params.allowZeroQuality = true;
            params.k = params.k / 5;
            params.classifier = MSTC.AuxClassifiers.ROT_2H;


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
