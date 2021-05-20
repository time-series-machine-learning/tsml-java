package tsml.classifiers.shapelet_based.dev.classifiers.ensemble;

import tsml.classifiers.TSClassifier;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstances;

import java.util.Random;

public class RandomEnsembleMSTC extends EnsembleMSTC {
    public RandomEnsembleMSTC(MSTC.ShapeletParams params) {
        super(params);
    }

    @Override //Abstract Ensemble
    public final void setupDefaultEnsembleSettings(TimeSeriesInstances data) {
        this.ensembleName = "ENS-MSTC";

        this.weightingScheme = new EqualWeightingTS();
        this.votingScheme = new MajorityConfidenceTS();

        this.numEnsembles = 100;
        TSClassifier[] classifiers = new MSTC[numEnsembles];

        Random r = new Random();

        MSTC.ShapeletQualities[] QUALITIES = MSTC.ShapeletQualities.values();
        MSTC.ShapeletFactories[] TYPES = MSTC.ShapeletFactories.values();


        for (int i=0;i<numEnsembles;i++){

            MSTC.ShapeletParams params = new MSTC.ShapeletParams(this.params);
            params.quality = QUALITIES[r.nextInt(QUALITIES.length)];
            params.type = TYPES[r.nextInt(TYPES.length)];
            if (params.type == MSTC.ShapeletFactories.DEPENDANT){
                params.compareSimilar = false;
            }

            classifiers[i] = new MSTC(params);
        }


        setClassifiers(classifiers);
    }
}
