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
package utilities;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class FileHandlingTools {

    public static void recursiveDelete(String directory) throws IOException {
        recursiveDelete(new File(directory));
    }
    
    public static void recursiveDelete(File directory) throws IOException {
        if (directory.isDirectory()) {
          for (File subDirectory : directory.listFiles())
            recursiveDelete(subDirectory);
        }
        Files.delete(directory.toPath());
//        if (!directory.delete()) 
//          throw new FileNotFoundException("Failed to delete file: " + directory);
    }
    
    /**
     * List the directories contained in the directory given
     */
    public static File[] listDirectories(String baseDirectory) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isDirectory();
            }
        });
    }

    /**
     * List the directories contained in the directory given
     */
    public static String[] listDirectoryNames(String baseDirectory) {
        return (new File(baseDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isDirectory();
            }
        });
    }
    
     /**
     * List the files contained in the directory given
     */
    public static File[] listFiles(String baseDirectory) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isFile();
            }
        });
    }
    
     /**
     * List the files contained in the directory given
     */
    public static String[] listFileNames(String baseDirectory) {
        return (new File(baseDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isFile();
            }
        });
    }
    
     /**
     * List the files contained in the directory given, that end with the given suffix (file extension, generally)
     */
    public static File[] listFilesEndingWith(String baseDirectory, String suffix) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isFile() && pathname.getName().endsWith(suffix);
            }
        });
    }
    
    /**
     * List the files contained in the directory given, that match the given regex
     */
    public static File[] listFilesMatchingRegex(String baseDirectory, String regex) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isFile() && pathname.getName().matches(regex);
            }
        });
    }
    
    /**
     * List the files contained in the directory given, that contain the given term
     */
    public static File[] listFilesContaining(String baseDirectory, String term) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isFile() && pathname.getName().contains(term);
            }
        });
    }
    
     /**
     * List the files contained in the directory given, that end with the given suffix (file extension, generally)
     */
    public static String[] listFileNamesEndingWith(String baseDirectory, String suffix) {
        return (new File(baseDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isFile() && name.endsWith(suffix);
            }
        });
    }
}
