package classifiers.distance_based.knn;

import classifiers.template.config.TemplateConfig;
import classifiers.template.config.reduced.NeighbourSearchStrategy;
import classifiers.template.config.reduced.ReductionConfig;
import classifiers.template.config.reduced.TrainEstimationSource;
import classifiers.template.config.reduced.TrainEstimationStrategy;
import classifiers.tuning.IterationStrategy;
import distances.DistanceMeasure;
import distances.time_domain.dtw.Dtw;
import utilities.ArrayUtilities;

import java.util.Comparator;

public class KnnConfig
    extends TemplateConfig {
    // config options
    private final static String K_KEY = "k";
    private int k = 1;
    private final static String DISTANCE_MEASURE_KEY = "dm";
    private DistanceMeasure distanceMeasure = new Dtw(0);
    private final static String EARLY_ABANDON_KEY = "ea";
    private boolean earlyAbandon = false;
    private final ReductionConfig reductionConfig = new ReductionConfig();

    public KnnConfig() {

    }

    public KnnConfig(KnnConfig other) throws
                                      Exception {
        super(other);
    }

    public static Comparator<KnnConfig> TRAIN_CONFIG_COMPARATOR = (config, other) -> { // todo
//            if(other.k < config.k) return 1;
//            else if(config.earlyAbandon != other.earlyAbandon) return 1; // todo dm
        return 0;
    };

    public static Comparator<KnnConfig> TEST_CONFIG_COMPARATOR = (config, other) -> { // todo
        return 0;
    };


    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
        ArrayUtilities.forEachPair(options, (key, value) -> {
            switch (key) {
                case K_KEY:
                    setK(Integer.parseInt(value));
                    break;
                case DISTANCE_MEASURE_KEY:
                    setDistanceMeasure(DistanceMeasure.fromString(value));
                    break;
                case EARLY_ABANDON_KEY:
                    setEarlyAbandon(Boolean.parseBoolean(value));
                    break;
            }
            return null;
        });
        distanceMeasure.setOptions(options);
        reductionConfig.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        return ArrayUtilities.concat(distanceMeasure.getOptions(), new String[] {
            DISTANCE_MEASURE_KEY,
            distanceMeasure.toString(),
            K_KEY,
            String.valueOf(k),
            EARLY_ABANDON_KEY,
            String.valueOf(earlyAbandon),
            }, reductionConfig.getOptions());
    }

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public boolean isEarlyAbandon() {
        return earlyAbandon;
    }

    public void setEarlyAbandon(final boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
    }

    @Override
    public KnnConfig copy() throws
                                   Exception {
        KnnConfig configuration = new KnnConfig();
        configuration.copyFrom(this);
        return configuration;
    }

    @Override
    public void copyFrom(final Object object) throws
                                              Exception {
        KnnConfig other = (KnnConfig) object;
        setOptions(other.getOptions());
        getReductionConfig().copyFrom(other.getReductionConfig());
    }

    public ReductionConfig getReductionConfig() {
        return reductionConfig;
    }
}
