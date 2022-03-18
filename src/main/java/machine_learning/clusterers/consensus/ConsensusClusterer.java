package machine_learning.clusterers.consensus;

import tsml.clusterers.EnhancedAbstractClusterer;

import java.util.ArrayList;

public abstract class ConsensusClusterer extends EnhancedAbstractClusterer {

    protected EnhancedAbstractClusterer[] clusterers;

    protected boolean buildClusterers = true;

    public ConsensusClusterer(EnhancedAbstractClusterer[] clusterers){
        this.clusterers = clusterers;
    }

    public ConsensusClusterer(ArrayList<EnhancedAbstractClusterer> clusterers){
        this.clusterers = clusterers.toArray(new EnhancedAbstractClusterer[0]);
    }

    public void setBuildClusterers(boolean b){
        buildClusterers = b;
    }
}
