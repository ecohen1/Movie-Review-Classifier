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
      for fFileObj in os.walk("reviews/movies_reviews"):
          lFileList = fFileObj[2]
          break
      print 'training on ' + str(len(lFileList)) + ' files'
      folds = 10
      for i in range(folds):

          self.goodReviewFrequency = {}
          self.badReviewFrequency = {}
          self.goodReviewFrequency["num_good_documents"] = 0
          self.badReviewFrequency["num_bad_documents"] = 0
        #   print lFileList[0:10]
          random.shuffle(lFileList)
          shuffledFileList = lFileList
        #   print shuffledFileList[0:10]
          endIndex = int(.9*len(shuffledFileList))
          trainFileList = shuffledFileList[0:endIndex]

          numRead = 0
          for review in trainFileList:
              reviewInfo = review.split('-')
              if len(reviewInfo) == 3:
                  numRead += 1
                  if reviewInfo[1] == '1':
                    #   print numRead
                      self.badReviewFrequency["num_bad_documents"] += 1
                      reviewText = self.loadFile("reviews/movies_reviews/"+review)
                      tokens = self.tokenize(reviewText)
                      for word in tokens:
                          if word in self.badReviewFrequency.keys():
                              self.badReviewFrequency[word] += 1
                          else:
                              self.badReviewFrequency[word] = 1
                  elif reviewInfo[1] == '5':
                    #   print numRead
                      self.goodReviewFrequency["num_good_documents"] += 1
                      reviewText = self.loadFile("reviews/movies_reviews/"+review)
                      tokens = self.tokenize(reviewText)
                      for word in tokens:
                          if word in self.goodReviewFrequency.keys():
                              self.goodReviewFrequency[word] += 1
                          else:
                              self.goodReviewFrequency[word] = 1

          self.save(self.goodReviewFrequency, "goodReviews"+str(i)+".txt")
          self.save(self.badReviewFrequency, "badReviews"+str(i)+".txt")
        #   print 'done training'
          self.CVtest(shuffledFileList, endIndex)

   def CVtest(self, fileArray, endIndex):
       testFileList = fileArray[endIndex:len(fileArray)]
       numFiles = 0
       numCorrectlyClassified = 0
    #    print testFileList
       for review in testFileList:
           tokenName = review.split("-")
           if len(tokenName) == 3:
               trueClass = tokenName[1]
               if trueClass == '1' or trueClass == '5':
                   numFiles += 1
                #    print 'classifying'
                   textToClassify = self.loadFile("reviews/movies_reviews/"+review)
                #    print trueClass, self.classify(textToClassify)
                   if trueClass == self.classify(textToClassify):
                       numCorrectlyClassified += 1

       percentageCorrect = float(numCorrectlyClassified/numFiles)
       print percentageCorrect


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
              wordProbabilityPositive = max(sys.float_info.min, (self.goodReviewFrequency[word] + 1)/(numGoodWords + (numGoodDocuments + numBadDocuments)))
              probabilityPositive += math.log(wordProbabilityPositive)
          else:
              wordProbabilityPositive = max(sys.float_info.min, 1/(numGoodWords + (numGoodDocuments + numBadDocuments)))
              probabilityPositive += math.log(wordProbabilityPositive)
          if word in self.badReviewFrequency.keys():
              wordProbabilityNegative = max(sys.float_info.min, (self.badReviewFrequency[word] + 1)/(numBadWords + (numGoodDocuments + numBadDocuments)))
              probabilityNegative += math.log(wordProbabilityNegative)
          else:
              wordProbabilityNegative = max(sys.float_info.min, 1/(numBadWords + (numGoodDocuments + numBadDocuments)))
              probabilityNegative += math.log(wordProbabilityNegative)
      probabilityPositive += math.log(max(sys.float_info.min, probabilityGoodDocument))
      probabilityNegative += math.log(max(sys.float_info.min, probabilityBadDocument))

      if probabilityPositive > probabilityNegative:
          return 5
      else:
          return 0


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
