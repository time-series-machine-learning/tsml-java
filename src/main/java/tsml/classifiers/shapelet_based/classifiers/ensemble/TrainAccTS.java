package tsml.classifiers.shapelet_based.classifiers.ensemble;

public class TrainAccTS extends ModuleWeightingSchemeTS {

    private double power = 1.0;

    public TrainAccTS() {
        uniformWeighting = true;
        needTrainPreds = true;
    }

    public TrainAccTS(double power) {
        this.power = power;
        uniformWeighting = true;
        needTrainPreds = true;
    }

    public double getPower() {
        return power;
    }

    public void setPower(double power) {
        this.power = power;
    }

    @Override
    public double[] defineWeighting(AbstractEnsembleTS.EnsembleModuleTS module, int numClasses) {
        return makeUniformWeighting(Math.pow(module.trainResults.getAcc(), power), numClasses);
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + (power==1.0 ? "" : "(" + power + ")");
    }

}
