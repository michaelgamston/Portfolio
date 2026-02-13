import spacy 
import json
import random
import re 
import datetime 
import pandas as pd
import pickle
import os
import sys
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QPushButton,QVBoxLayout, QWidget, QLabel, QScrollArea, QSizePolicy, QHBoxLayout

class ResponseTile(QWidget):
    """A widget to display a response

    Args:
        QWidget (PyQt Widget): A widget from PyQt
    """
    
    def __init__(self, input, is_user):
        super().__init__()
        
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        #create a layout for the response title and response
        layout = QVBoxLayout()

        #check which type of user is using the serive 
        if is_user:
            title = QLabel('You')
        else: 
            title = QLabel('Marvin')

        #create a layout to place the response label in 
        response_layout = QHBoxLayout()      

        #create a label to put the response in
        input_label = QLabel()
        input_label.setText(input)
        #make sure the words stay within the frame
        input_label.setWordWrap(True) 

        #set style rules based on who the response is from 
        if is_user:
            #set a title and style for if the response is from the user
            input_label.setObjectName("user")
            input_label.setStyleSheet(f"""
                QLabel#user {{
                    background-color: #14e02c; 
                    border-radius: 15px; 
                    padding: 10px;
                }}
            """)
        else:
            #set a title and style for if the response is from marvin
            input_label.setObjectName("marvin")
            input_label.setStyleSheet(f"""
                QLabel#marvin {{
                    background-color: #4aa4d9;
                    border-radius: 15px; 
                    padding: 10px;
                }}
            """)

        if is_user:
            #if its the user responding put the tile on the right hand side
            title.setAlignment(Qt.AlignRight)
            layout.addWidget(title)
            response_layout.addStretch()
            response_layout.addWidget(input_label)
            
        else:
            #if its marvin set the tile from the left
            layout.addWidget(title)
            response_layout.addWidget(input_label)
            response_layout.addStretch()

        #add the response under the title
        layout.addLayout(response_layout)
        #add it to the lay out
        self.setLayout(layout)

class Chatbot(QMainWindow):
    """The main class for displaying the chatbot and producing outputs

    Args:
        QMainWindow (PyQt type): PyQt main window class
    """
    
    def __init__(self):
        super().__init__()
        # add a title
        self.setWindowTitle("Marvin Medical Helper")
        # set the size of the output
        self.setGeometry(100, 100, 600, 600)

        #container widget to encase the everything
        container = QWidget()
        #se it centrally 
        self.setCentralWidget(container)
        #add a layout to add subsequent
        layout = QVBoxLayout()

        #create a scrollable window so the whole of the chat is visable 
        self.scroll_area = QScrollArea()
        #allow it change size as more data is added to it 
        self.scroll_area.setWidgetResizable(True)

        #create a widget to hold the chat boxes - add a layout for further chat to be added
        self.chat_display= QWidget()
        self.chat_layout = QVBoxLayout(self.chat_display)
        #add a strech so the layout fills the whole screen 
        self.chat_layout.addStretch()
        #add it to the scroll area
        self.scroll_area.setWidget(self.chat_display)
        #add the scroll area to overall layout
        layout.addWidget(self.scroll_area)

        #add a box for user input 
        self.text_input = QLineEdit(self)
        #if enter is pressed activate the run marving to get a response
        self.text_input.returnPressed.connect(self.run_marvin)
        layout.addWidget(self.text_input)

        #add a send button to activate marvin
        send_button = QPushButton("Send", self)
        send_button.clicked.connect(self.run_marvin)
        layout.addWidget(send_button)
        container.setLayout(layout)
        
        #knowledge tracker
        self.knowledge = { 
                    'name' : 'UNKNOWN',
                    'D.O.B' : 'UNKNOWN',
                    'intent' : 'UNKNOWN',
                    'handle_intent' : 'UNKNOWN',
                    'carry_out_request' : 'UNKNOWN',
                    }
        #store user id once found
        self.user_id = None
        
        
        #create a json file to holding requests to change data in pateint files 
        if not os.path.isfile("changes_for_review.json"):
            with open("changes_for_review.json", "w") as f:
                dictonary = {"reviews" : []}
                json.dump(dictonary, f)
        
        #tokeniser 
        self.nlp = spacy.load("en_core_web_sm")
        
        #health database the chatbot will interact with
        self.health_data = pd.read_csv('data/processed_health_data.csv')
        
        #bag of words vectoriser 
        with open("models/vectoriser.pkl", "rb") as f:
            self.vectoriser = pickle.load(f)
        
        #SVM model for intents and the reponses 
        with open("models/intent_model.pkl", "rb") as f:
            self.intent_model = pickle.load(f)
        with open("responses/intent_responses.json", "r") as f:
            self.intent_responses = json.load(f)
        
        #SVM model for the date of birth section and the responses 
        with open("models/unrecognsied_input_DOB_model.pkl", "rb") as f:
            self.dob_input_model = pickle.load(f)
        with open("responses/dob_responses.json", "r") as f:
            self.dob_responses = json.load(f)
        
        #SVM model for the name section and the responses
        with open("models/unrecognsied_input_name_model.pkl", "rb") as f: 
            self.name_input_model = pickle.load(f)
        with open("responses/name_responses.json", "r") as f:
            self.name_responses = json.load(f)
            
        #SVM model for handle_intent and reponses 
        with open("models/handle_intent_model.pkl", "rb") as f: 
            self.handle_intent_model = pickle.load(f)
        with open("responses/handle_intent_responses.json", "r") as f: 
            self.handle_intent_responses = json.load(f)
        
        #general greetings file
        with open("responses/greetings.json", "r") as f:
            self.greetings = json.load(f)
            
        with open("responses/switch_user_responses.json", "r") as f:
            self.switch_user_responses = json.load(f)
        
        #flags to change the running of marvin 
        self.first_task_complete = False
        self.update_loop = False
        
        #print an intial message 
        self.give_response(random.choice(self.greetings['greetings']))

    def run_marvin(self):
        """Once an input is generated this function runs marvin to find what to do next.
        """
        
        user_input = self.text_input.text().strip()
        
        
        if user_input:
            #Create an instance of the response tile with the input
            response = ResponseTile(f"{user_input}", True)
            #add it to the bottom of the chat layout
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, response)
            
            #find the state so marvin can response properly 
            state = self.find_state()
            
            if state == 'name':
                self.find_name(user_input)
            elif state == 'D.O.B':
                self.find_dob(user_input)
            elif state == 'intent':
                self.get_intent(user_input) 
            elif state == 'handle_intent':
                self.handle_intent(user_input)
            elif state == 'carry_out_request':
                if self.update_loop:
                    #replay this section untill all the update requests have been cleared 
                    self.handle_edit_request(user_input)
                else:
                    self.carry_out_request(user_input)
                
                self.first_task_complete = True
                
            #find is a response from marvin is need after thread complete
            self.get_next_instruction()

            #clear the input
            self.text_input.clear()

            #wait untill marvins responses is displayed and then scroll to the bottom
            chat_scroll = self.scroll_area.verticalScrollBar()
            QtCore.QTimer.singleShot(100, lambda: chat_scroll.setValue(chat_scroll.maximum()))
        
    def get_next_instruction(self):
        """A function to find if a response is needed based on the input
        """
        
        state = self.find_state()
        
        if state == 'name':
            self.give_response(random.choice(self.name_responses['request_name']))
        elif state == 'D.O.B':
            self.give_response(random.choice(self.dob_responses['request_dob']))
        elif state == 'intent':
            if self.first_task_complete: 
                self.give_response(random.choice(self.intent_responses['post_runthrough_request']))
            else: 
                self.give_response(random.choice(self.intent_responses['first_request']))
            
    def give_response(self, resposnse):
        """A function to display a response from marvin

        Args:
            resposnse (_type_): _description_
        """
        
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, ResponseTile(f"{resposnse}", False))

    def find_name(self, user_input):
        """A function to handle the name state.
        This function extracts the name of user from their input

        Args:
            user_input (string): the users input
        """
        
        #process the input
        processed_text, meta_data, text_vector = self.preprocess_text(user_input)
        
        #check if they want to exit or help
        if processed_text[0] == "exit": 
            self.close_app()
            return   
        elif processed_text[0] == "help":
            self.give_response(random.choice(self.name_responses["help"]))
            return
    
            
        #run the text vector through the SVM
        intent = self.name_input_model.predict(text_vector)[0]
        
        try:
            if (len(meta_data['names']) > 1):
                #more than one name found
                self.check_response(intent, "name")
            else:
                #one name found
                names = meta_data['names'][0].split()
                if len(names) >= 2: 
                    first_name = names[0]
                    last_name = names[-1]
                    #check to see if we have that name on record 
                    if "".join([first_name," ",last_name]) in self.health_data["Name"].tolist(): 
                        self.knowledge['name'] = {'first_name' : first_name, 'last_name' : last_name}
                        self.give_response(f"Thanks {first_name}.")
                    else: 
                        #if name not on record
                        self.give_response(random.choice(self.name_responses['name_not_found']))
                    
                else: 
                    self.check_response(intent, "name")
   
        except: 
            #no names found
            self.check_response(intent, "name")
            
    def find_dob(self, user_input):
        """A function to handle the date of birth state.
        This function extracts the date of birth of the user from their input

        Args:
            user_input (string): the users input
        """
        
     
            
        #retreive and process user data
        processed_text, meta_data, text_vector = self.preprocess_text(user_input)
        
        #check if they want to exit or help
        if processed_text[0] == "exit": 
            self.close_app()
            return
        elif processed_text[0] == "help":
            self.give_response(random.choice(self.dob_responses["help"]))
            return
                            
        #run the text vector through the SVM
        intent = self.dob_input_model.predict(text_vector)[0]
        
        
        if (len(meta_data['dates']) > 1) or (len(meta_data['dates']) == 0): 
        # if too many dates found or no date found 
            #check the context of the input
            self.check_response(intent, "dob")
        else: 
            #if a valid date has been given 
            #calculate their age
            dob = meta_data['dates'][0]
            age = self.calculate_age(dob)
            name = "".join([self.knowledge["name"]["first_name"], " ", self.knowledge["name"]["last_name"]])
            
            #check if their age correlate with their name
            if self.health_data.loc[self.health_data['Name'] == name, 'Age'].iloc[0] == age: 
                self.knowledge['D.O.B'] = age
                self.user_id = self.health_data[self.health_data['Name'] == name].index[0]
                self.give_response("Great, I've found your records")
            else: 
                self.give_response("It seems that date of birth doesn't match what I have on record.")
        

    
    def get_intent(self, user_input):
        """A function to handle the intent state.
        This function extracts what the user would like to do with their records.
        It can also extract which records the user wants to effect, if they are provided at this stage.

        Args:
            user_input (string): the users input
        """

        #retreive and process user data
        processed_text, meta_data, text_vector = self.preprocess_text(user_input)
        
        #check if they want to exit
        if processed_text[0] == "exit": 
            self.close_app()
            return
        elif processed_text[0] == "help": 
            self.give_response(random.choice(self.intent_responses["help"]))
            return
        #run processed_text through intent svm 
        intent = self.intent_model.predict(text_vector)[0]
        
        #find which intent the classifer has identified and respond 
        if intent == "other": 
            self.give_response(random.choice(self.intent_responses['other']))
            return
        elif intent == "question":
            self.give_response(random.choice(self.intent_responses['question_response']))
            return 
        elif intent == "get_record":
            self.give_response("Sure I can retrieve record(s) for you")
            self.knowledge["intent"] = "get_record"    
        elif intent == "delete_record": 
            self.give_response("Sure I can delete record(s) for you")
            self.knowledge["intent"] = "delete_record"    
        elif intent == "update_record":
            self.give_response("Sure I can update record(s) for you")
            self.knowledge["intent"] = "update_record"
        else:
            self.give_response(random.choice(self.intent_responses['other'])) 
        
        #find the data points in the user response if they are included
        insterested_data_points = self.find_interested_datapoints(processed_text)
        
        if len(insterested_data_points) > 0: 
            #if they have given datapoints add them to the knowledge
            self.knowledge['handle_intent'] = insterested_data_points 
            self.give_response(f"""Can you confirm you want to affect the following data points: {self.knowledge['handle_intent']}""")
        else: 
            #if no datapoints given ask them which ones they want
            self.give_response(random.choice(self.intent_responses["ask_for_data"]))
        
            
    def handle_intent(self, user_input): 
        """Once an intent has been found, this function handles carring out the request 

        Args:
            intent (string): the intent type
            processed_text (list): the users previous input
            insterested_data_points (list): possible targets for the request
        """
        processed_text, meta_data, text_vector = self.preprocess_text(user_input)
        
        #check for help, exit or change request keyword
        if processed_text[0] == "exit": 
            self.close_app()
            return
        if processed_text[0] == "help":
            self.give_response(random.choice(self.handle_intent_responses["help_with_datapoints"]))
            return
        if (processed_text[0] == "change") and (processed_text[1] == "request"):
            self.reset()
            return
            
        #find the data points in the user response 
        insterested_data_points = self.find_interested_datapoints(processed_text)
     
       
        if len(insterested_data_points) > 0: 
            #if they still haven't given any datapoints find out why and self.give_response a response
            self.knowledge['handle_intent'] = insterested_data_points  
            self.give_response(f"""Can you confirm you want to affect the following data points: {self.knowledge['handle_intent']}""")
        
        else:
            #if they still haven't given any datapoints find out why and self.give_response a response
            outcome = self.handle_intent_model.predict(text_vector)
            self.check_response(outcome, "handle_intent")
            #give intent related questions
            if self.knowledge["intent"] == "delete_record":
                self.give_response("Which record/s would you like to delete")
                
            if self.knowledge["intent"] == "get_record":
                self.give_response("Which record/s would you like retreive?")
                
            if self.knowledge["intent"] == "update_record":
                self.give_response("Which record/s would you like to request to change?")
            
        
    def carry_out_request(self, user_input):
        """Once an intent has been identified with target datapoints this function carries out the request 

        Args:
            user_input (string): the users input
        """
        
        deletable_datapoints = ['Gender', 'Blood Type', 'Medical Condition',
       'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider',
       'Billing Amount', 'Room Number', 'Admission Type', 'Discharge Date',
       'Medication', 'Test Results']
        
        #retreive and process user data
        processed_text, meta_data, text_vector = self.preprocess_text(user_input)
        
        #the user is asked if they are sure they want to proceed before this function is called 
        #this checks if the user wants to continue or not 
        outcome = self.yes_no_classifier(processed_text)
        
        
        if outcome == 'positive':
            
            #if positive find which intent was identified
            if self.knowledge['intent'] == "delete_record":
                
                #delete the requested records
                for column in self.knowledge['handle_intent']:
                    if column in deletable_datapoints:
                        self.health_data.at[self.user_id, column] = pd.NA
                        self.give_response(f"Column {column} has been deleted for you.")
                    else:
                        self.give_response(f"I cannot delete {column} as it's need for reference. You can request to change it.")
                
                self.health_data.to_csv("data/processed_health_data.csv", index=False)
                
                #self.give_response('Thats all done for you, is there anything else I can help with?')
                self.reset()
                
                return 
                
            if self.knowledge['intent'] == "get_record":
                #print out the selected records
                for column in self.knowledge['handle_intent']:
                    record = self.health_data.at[self.user_id, column]
                    self.give_response(f"{column} : {record}")   
                #self.give_response('Thats all done for you, is there anything else I can help with?')
                self.reset()
                return
                
            if self.knowledge['intent'] == "update_record":
                #set the update_loop flag to true 
                #the user will be asked what change they want to make for each datapoint they/ve requested 
                self.give_response(f"What would you like to have changed for {self.knowledge['handle_intent'][0]}?")
                self.update_loop = True
                return 
        #if negative reset the datapoints so the user can choose new ones
        elif outcome == 'negative':
            self.knowledge['handle_intent'] = 'UNKNOWN'
            self.give_response("What would you like to effect?")
        else:
        #if neither a positive or negative find out why, give response and go back to request
            intent = self.handle_intent_model.predict(text_vector)
            if processed_text[0] == "exit": 
                self.close_app()
                return
            elif processed_text[0] == "help":
                self.give_response(random.choice(self.handle_intent_responses["help_with_datapoints"]))
                return
            elif (processed_text[0] == "change") and (processed_text[1] == "request"):
                self.reset()
                return
            
            self.check_response(intent, "handle_intent")
            self.give_response(f"""Can you confirm you want to affect the following data points: {self.knowledge['handle_intent']}""")
        
    def handle_edit_request(self,user_input):
        """Handles creating change requests 
        The functions pops the first element out of the handle_intent list and asks the user what they would like to change 

        Args:
            user_input (string): user input 
        """
        changes = {}
            
        processed_text, meta_data, text_vector = self.preprocess_text(user_input)
        
        #check for key words
        if processed_text[0] == "exit": 
            self.close_app()
            return 
        elif processed_text[0] == "help":
            self.give_response("Please enter the changes you would like to request.")
            return
        elif (processed_text[0] == "change") and (processed_text[1] == "request"):
            self.reset()
            return

        #retrieve the first datapoint to change
        target = self.knowledge['handle_intent'].pop(0)
        
        #store what datapoint it is and what the user wants to change it to
        changes[target] = user_input
        
        #create a dictionary to contain the patients id and the changes
        tagged_changes = {"patient" : str(self.user_id), "changes" : changes}
        
        #add the changes to a json for the "doctor" to review 
        with open("changes_for_review.json", "r+") as f:
            changes_for_review = json.load(f)
            f.seek(0)
            changes_for_review["reviews"].append(tagged_changes) 
            json.dump(changes_for_review, f)
            f.truncate()  
            
        #keep asking for changes untill handle_intent list is empty and the request have been fulfilled
        if len(self.knowledge['handle_intent']) > 0:    
            self.give_response(f"What would you like to have changed for {self.knowledge['handle_intent'][0]}?")
        else: 
            self.give_response("A request has been sent with your changes. Your Doctor will review it and get back to you. Is there anything else I can help with?")
            self.reset()
            self.update_loop = False
            
    def reset(self, change_user = False):
        """This fucntion resest user data

        Args:
            change_user (bool, optional): True - total reset. False - just reset 
        """
        
        if change_user: 
            self.knowledge = { 
                        'name' : 'UNKNOWN',
                        'D.O.B' : 'UNKNOWN',
                        'intent' : 'UNKNOWN',
                        'handle_intent' : 'UNKNOWN',
                        'carry_out_request' : 'UNKNOWN',
                        }
            
            self.user_id = None    
        else: 
            self.knowledge['intent'] = 'UNKNOWN'
            self.knowledge['handle_intent'] = 'UNKNOWN'
            self.knowledge['carry_out_request'] = 'UNKNOWN'
    
    def find_interested_datapoints(self, processed_text):
        """Find which column titles have been mention by the user

        Args:
            processed_text (list): all words in the users in put

        Returns:
            list : a list of columns selected by the user
        """
        
        #the columns following processing
        processed_text_matches = ['name', 'age', 'gender', 'blood type', 'medical condition',
        'date admission', 'doctor', 'hospital', 'insurance provider',
        'billing amount', 'room number', 'admission type', 'discharge date',
        'medication', 'test result']
        
        #the actual columns titles in pandas dataframe
        pandas_columns = ['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition',
        'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider',
        'Billing Amount', 'Room Number', 'Admission Type', 'Discharge Date',
        'Medication', 'Test Results']

        #a list for the column titles found
        insterested_data_points = []
        
        #check if they want all of them
        if any([x in processed_text for x in ['all', 'everything']]):
            return pandas_columns
        else:
            #if not all of them then find the ones they want, and add the corrisponding dataframe title to the data points list
            for index, words in enumerate(processed_text_matches):
                split_words = words.split()
                if set(split_words).issubset(processed_text):
                    insterested_data_points.append(pandas_columns[index])
            
                
        return insterested_data_points            
                             
    def yes_no_classifier(self, processed_text): 
        """A function to find if the user has reponsed negatively or positively

        Args:
            processed_text (list): user input

        Returns:
            string: positive or negative output
        """
        #negative responses 
        negative = ['no', 'nope', 'nah','not', 'dont', 'do not', 'decline']
        
        #positive responses
        positive = ['yep', 'yeah', 'yes', 'do it', 'please', 'accept']
        
        #search for negative responses 
        for word in negative: 
            if set(word.split()).issubset(processed_text):
                return 'negative'
        
        #search for positive responses 
        for word in positive: 
            if set(word.split()).issubset(processed_text):
                return 'positive'
            
        #if nothing found return false 
        return False
                      
    def check_response(self, intent, section):
        """checks the output from the SVMs and self.give_responses responses

        Args:
            intent (string): the identified response 
            section (string): the section where the intent was idenitied
        """
        #match intent with responses 
        if section == "name":
            responses = self.name_responses
        elif section == "dob":
            responses = self.dob_responses
        elif section == "handle_intent":
            responses = self.handle_intent_responses  
        
        #respond 
        if intent == "question": 
            self.give_response(random.choice(responses['question_response']))
        elif intent == "other": 
            self.give_response(random.choice(responses['other']))
        elif intent == "change_user": 
            self.knowledge = self.knowledge = { 
                        'name' : 'UNKNOWN',
                        'D.O.B' : 'UNKNOWN',
                        'intent' : 'UNKNOWN',
                        'handle_intent' : 'UNKNOWN',
                        'carry_out_request' : 'UNKNOWN',
                        }
            self.give_response(random.choice(self.switch_user_responses["user_reset"]))
    
                
   
    def find_state(self): 
        """ runs at the start of the loop to find what parts of knowledge are still unknown 

        Returns:
            string: state
        """
        
        #find the unknown keys 
        for key in self.knowledge.keys(): 
            if self.knowledge[key] == 'UNKNOWN': 
                return key
        
    def represent_text_bow(self, processed_text):
        """vectorise tokens into a Bag Words formate 

        Args:
            processed_text (list): a list of words that have been tokenised by spacy

        Returns:
            vector: vectorisied text
        """
        # Initalise CountVectoriser 
        processed_sentence = " ".join(processed_text) #join tokens back to string for vectoriser input
        #vectorise the processed sentence 
        text_vector = self.vectoriser.transform([processed_sentence])
        return text_vector     
    
    def preprocess_text(self, text):
        """Tokensise user input
        Turns tokens into bag of words 
        Then extracts any meta data - names and dates
        

        Args:
            text (string): user input

        Returns:
            processed_tokens: the tokens from spacy
            meta_data: dictionary contianing the meta date
            text_vector: bag of words vector
        """
        
        #
        doc = self.nlp(text)
        processed_tokens = []
        meta_data = {}

        #tokenise string and append it to a list ready for further processing
        for token in doc: 
            if (not token.is_punct):
                processed_tokens.append(token.lemma_.lower())
        
        if len(processed_tokens) == 0: 
            processed_tokens.append(" ")        
        
        text_vector = self.represent_text_bow(processed_tokens).toarray()
        
        #extract any names if they are present 
        try:
            names = []
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    names.append(ent.text.lower())
                    
            if len(names) > 0:        
                meta_data['names'] = names
     
        except: 
            pass

     
        dates = []
        #looks for all versions of a date with a string for the month
        dates.extend(re.findall(r'(?:\d{1,2} )?(?:jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)[a-z]* (?:\d{1,2}, )?\d{2,4}',text)) 
        #looks for all versions of the data in int format
        dates.extend(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',text))
        meta_data['dates'] = dates 
                
        
        return processed_tokens, meta_data, text_vector
    
    def calculate_age(self, dob):
        """calculates the age of the user based on their date of birth

        Args:
            dob (string): date of birth

        Returns:
            int: the users age
        """
        
        date_code = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']                
        day, _, year = split_dates = dob.split()
        
        #if the month is a string
        if len(split_dates[1]) >= 3: 
            month = date_code.index(split_dates[1][:3])+1
        #if the month is a int
        else: 
            month = int(split_dates[1])
            
        #calculate age (age is used instead of date of birth in the databaes I am using)
        current_date = datetime.datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        current_day = current_date.day
        
        #if its not thier birth month 
        if int(month) < current_month: 
            age = (current_year - int(year))  
        #if it is their birth month 
        elif int(month) == current_month:
            #if it isn't yet their birthday 
            if int(day) <= current_day: 
                age = (current_year - int(year)) 
            #if it is or has been past thier birthday 
            else: 
                age = current_year - int(year) - 1
        #if it's past their birth month
        else: 
            age = current_year - int(year) - 1

        return age
    
    def close_app(self):
        """only use before closing so the system hands for a second to print
        """
        self.knowledge = {"wait" : "wait"}
        self.give_response(random.choice(self.greetings["goodbye"]))
        QtCore.QTimer.singleShot(2000, self.close)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    chatbot_app = Chatbot()
    chatbot_app.show()
    sys.exit(app.exec_())