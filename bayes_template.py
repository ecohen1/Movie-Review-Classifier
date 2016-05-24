# NetIDs: etc025, wom020
# Name: Eli Cohen, Wajihuddin Omar Mohammed
# Date: 5/23/16
# Description: Homework 4
#
#

import math, os, pickle, re, random, sys

class Bayes_Classifier:

   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
      cache of a trained classifier has been stored, it loads this cache.  Otherwise,
      the system will proceed through training.  After running this method, the classifier
      is ready to classify input text."""
      self.goodReviewFrequency = {}
      self.badReviewFrequency = {}

      if os.path.isfile('./goodReviews.txt') and os.path.isfile('./badReviews.txt'):
          self.goodReviewFrequency = self.load("goodReviews.txt")
          self.badReviewFrequency = self.load("badReviews.txt")
      else:
          self.train()


   def train(self):
      """Trains the Naive Bayes Sentiment Classifier."""

      lFileList = []
      for fFileObj in os.walk("movies_reviews"):
          lFileList = fFileObj[2]
          break
      random.shuffle(lFileList)
      shuffledFileList = lFileList[0:len(lFileList)]
      print 'training on ' + str(len(shuffledFileList)) + ' files'

      folds = 10
      for i in range(folds):

          self.goodReviewFrequency = {}
          self.badReviewFrequency = {}
          self.goodReviewFrequency["num_good_documents"] = 0
          self.badReviewFrequency["num_bad_documents"] = 0
        #   print lFileList[0:10]
        #   print shuffledFileList[0:10]
          startIndex = int(len(shuffledFileList)*float(i)/folds)
          endIndex = int(len(shuffledFileList)*float(i+1)/folds)
          print startIndex, endIndex
          trainFileList = shuffledFileList[0:startIndex]
          trainFileList.extend(shuffledFileList[endIndex:len(shuffledFileList)])
          testFileList = shuffledFileList[startIndex:endIndex]
          print len(trainFileList), len(testFileList)

          numRead = 0
          for review in trainFileList:
              reviewInfo = review.split('-')
              if len(reviewInfo) == 3:
                  numRead += 1
                  if reviewInfo[1] == '1':
                    #   print numRead
                      self.badReviewFrequency["num_bad_documents"] += 1
                      reviewText = self.loadFile("movies_reviews/"+review)
                      tokens = self.tokenize(reviewText)
                      for word in tokens:
                          if word in self.badReviewFrequency.keys():
                              self.badReviewFrequency[word] += 1
                          else:
                              self.badReviewFrequency[word] = 1
                  elif reviewInfo[1] == '5':
                    #   print numRead
                      self.goodReviewFrequency["num_good_documents"] += 1
                      reviewText = self.loadFile("movies_reviews/"+review)
                      tokens = self.tokenize(reviewText)
                      for word in tokens:
                          if word in self.goodReviewFrequency.keys():
                              self.goodReviewFrequency[word] += 1
                          else:
                              self.goodReviewFrequency[word] = 1

          self.save(self.goodReviewFrequency, "goodReviews"+str(i)+".txt")
          self.save(self.badReviewFrequency, "badReviews"+str(i)+".txt")
          precision, recall, f_measure = self.crossValidation(testFileList)
          f = open('results.txt', 'a')
          f.write(str(i) + " trial resulted in " + str(precision) + ", " + str(recall) + ", " + str(f_measure) + " precision, recall, and f_measure\n")
          f.close()

   def crossValidation(self, testFileList):
       numFiles = 0
       truePositives = 0
       trueNegatives = 0
       falsePositives = 0
       falseNegatives = 0

    #    print testFileList
       for review in testFileList:
           tokenName = review.split("-")
           if len(tokenName) == 3:
               trueClass = tokenName[1]
               if trueClass == '1' or trueClass == '5':
                   numFiles += 1
                #    print 'classifying'
                   textToClassify = self.loadFile("movies_reviews/"+review)
                   classification = self.classify(textToClassify)
                #    print trueClass, classification
                   if trueClass == classification:
                       if trueClass == '1':
                           trueNegatives += 1
                        #    print 'trueNegative'
                       else:
                           truePositives += 1
                        #    print 'truePositive'
                   else:
                       if trueClass == '1':
                           falsePositives += 1
                        #    print 'falsePositive'
                       else:
                           falseNegatives += 1
                        #    print 'falseNegative'

    #    percentageCorrect = float(numCorrectlyClassified)/numFiles
       if truePositives + falsePositives != 0:
           precision = float(truePositives)/(truePositives + falsePositives)
       else:
           precision = 0.0
       if truePositives + falseNegatives:
           recall = float(truePositives)/(truePositives + falseNegatives)
       else:
           recall = 0.0
       if precision != 0 or recall != 0:
           f_measure = 2*precision*recall/(precision + recall)
       else:
           f_measure = 0.0
       return (precision, recall, f_measure)


   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      probabilityPositive = 0
      probabilityNegative = 0

      numGoodDocuments = self.goodReviewFrequency["num_good_documents"]
      numBadDocuments = self.badReviewFrequency["num_bad_documents"]
      probabilityGoodDocument = numGoodDocuments/(numGoodDocuments + numBadDocuments)
      probabilityBadDocument = numBadDocuments/(numGoodDocuments + numBadDocuments)
      numGoodWords = 0
      numBadWords = 0
      for goodWord in self.goodReviewFrequency.keys():
          numGoodWords += self.goodReviewFrequency[goodWord]
      for badWord in self.badReviewFrequency.keys():
          numBadWords += self.badReviewFrequency[badWord]

      tokens = self.tokenize(sText)
      for word in tokens:
          if word in self.goodReviewFrequency.keys():
              wordProbabilityPositive = max(sys.float_info.min, (self.goodReviewFrequency[word] + 1.0)/(numGoodWords + (numGoodDocuments + numBadDocuments)))
              probabilityPositive += math.log(wordProbabilityPositive)
          else:
              wordProbabilityPositive = max(sys.float_info.min, 1.0/(numGoodWords + (numGoodDocuments + numBadDocuments)))
              probabilityPositive += math.log(wordProbabilityPositive)
          if word in self.badReviewFrequency.keys():
              wordProbabilityNegative = max(sys.float_info.min, (self.badReviewFrequency[word] + 1.0)/(numBadWords + (numGoodDocuments + numBadDocuments)))
              probabilityNegative += math.log(wordProbabilityNegative)
          else:
              wordProbabilityNegative = max(sys.float_info.min, 1.0/(numBadWords + (numGoodDocuments + numBadDocuments)))
              probabilityNegative += math.log(wordProbabilityNegative)
      probabilityPositive += math.log(max(sys.float_info.min, probabilityGoodDocument))
      probabilityNegative += math.log(max(sys.float_info.min, probabilityBadDocument))

      if probabilityPositive > probabilityNegative:
          return '5'
      else:
          return '1'


   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt

   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()

   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText):
      """Given a string of text sText, returns a list of the individual tokens that
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))

      if sToken != "":
         lTokens.append(sToken)

      return lTokens
