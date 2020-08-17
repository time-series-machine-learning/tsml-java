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
package tsml.classifiers;

import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import utilities.FileUtils;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Interface that allows the user to allow a classifier to checkpoint, i.e. 
save its current state and then load it again to continue building the model on 
a separate run.

By default this involves simply saving and loading a serialised the object 

known classifiers: none

Requires two methods 
number 

 * @author Tony Bagnall 2018, goastler
 */
public interface Checkpointable extends Serializable {

    /**
     * Store the path to write checkpoint files,
     * @param path string for full path for the directory to store checkpointed files
     * @return true if successful (i.e. the directory now exist
     */
    boolean setCheckpointPath(String path);

    /**
     * DEFAULT FOR NOW: make abstract when fully implemented
     * @param t number of hours between checkpoints
     * @return true if set correctly.
     */
    default boolean setCheckpointTimeHours(int t){ return false;};

    //Override both if not using Java serialisation
    default void saveToFile(String filename) throws Exception {
        try (FileUtils.FileLock fileLocker = new FileUtils.FileLock(filename);
             FileOutputStream fos = new FileOutputStream(fileLocker.getFile());
             GZIPOutputStream gos = new GZIPOutputStream(fos);
             ObjectOutputStream out = new ObjectOutputStream(gos)) {
            out.writeObject(this);
        }
    }
    default void loadFromFile(String filename) throws Exception{
        Object obj = null;
        try (FileUtils.FileLock fileLocker = new FileUtils.FileLock(filename);
             FileInputStream fis = new FileInputStream(fileLocker.getFile());
             GZIPInputStream gis = new GZIPInputStream(fis);
             ObjectInputStream in = new ObjectInputStream(gis)) {
            obj = in.readObject();
        }
        if(obj != null) {
            copyFromSerObject(obj);
        }
    }



    /**
     * Utility function to set the file structure up if required. Call this in setSavePath if you wish
     * */
    default boolean createDirectories(String path){
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }

    //Define how to copy from a loaded object to this object
    void copyFromSerObject(Object obj) throws Exception;

}
