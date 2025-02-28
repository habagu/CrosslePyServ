import json

def Puzzle_to_JsonArray(Puzzle):
    JsonString = ""
    id = 0
    JsonArray = []
    for cell in Puzzle:
        id = id + 1
        if cell["arrow_to_bottom"] == True:
            x = cell["x"]
            y = cell["y"] + 1
            text = [e.get("text_value") for e in Puzzle if e.get("x") == x and e.get("y") == y]
            tempy = y - 1
            tempx = x
            length = 0
            found = any(e["x"] == tempx and e["y"] == tempy and e["text"] == False for e in Puzzle)
            slots = []
            while found:
                found = any(e["x"] == cell["x"] and e["y"] == tempy and e["text"] == False for e in Puzzle)
                slots.append({str(tempx)+":"+str(tempy)})
                length = length + 1
                tempy = tempy + 1
            element = {
                        "id": id,
                        "question": text,
                        "position": str(x) + ":" + str(y), #Position der Frage im Grid
                        "slots": slots, #Positionen der Zeichen der Antwort im Grid
                        "length": length
                    }
            JsonArray.append(element)
        elif cell["arrow_to_right"] == True:
            x = cell["x"] + 1
            y = cell["y"]
            text = [e.get("text_value") for e in Puzzle if e.get("x") == x and e.get("y") == y]
            tempx = x - 1
            tempy = y
            length = 0
            found = any(e["x"] == tempx and e["y"] == tempy and e["text"] == False for e in Puzzle)
            slots = []
            while found:
                found = any(e["x"] == cell["x"] and e["y"] == tempy and e["text"] == False for e in Puzzle)
                slots.append({str(tempx)+":"+str(tempy)})
                length = length + 1
                tempx = tempx + 1
            element = {
                        "id": id,
                        "question": text,
                        "position": str(x) + ":" + str(y), #Position der Frage im Grid
                        "slots": slots, #Positionen der Zeichen der Antwort im Grid
                        "length": length
                    }
            JsonArray.append(element)
        elif cell["handle_bottom_to_right"]== True:
            x = cell["x"]
            y = cell["y"] - 1
            text = [e.get("text_value") for e in Puzzle if e.get("x") == x and e.get("y") == y]
            tempx = x 
            tempy = y + 1
            length = 0
            found = any(e["x"] == tempx and e["y"] == tempy and e["text"] == False for e in Puzzle)
            slots = []
            while found:
                found = any(e["x"] == cell["x"] and e["y"] == tempy and e["text"] == False for e in Puzzle)
                slots.append({str(tempx)+":"+str(tempy)})
                length = length + 1
                tempx = tempx + 1
            element = {
                        "id": id,
                        "question": text,
                        "position": str(x) + ":" + str(y), #Position der Frage im Grid
                        "slots": slots, #Positionen der Zeichen der Antwort im Grid
                        "length": length
                    }
            JsonArray.append(element)
        elif cell["handle_left_to_bottom"]== True:
            x = cell["x"] - 1
            y = cell["y"]
            text = [e.get("text_value") for e in Puzzle if e.get("x") == x and e.get("y") == y]
            tempx = x + 1
            tempy = y
            length = 0
            found = any(e["x"] == tempx and e["y"] == tempy and e["text"] == False for e in Puzzle)
            slots = []
            while found:
                found = any(e["x"] == cell["x"] and e["y"] == tempy and e["text"] == False for e in Puzzle)
                slots.append({str(tempx)+":"+str(tempy)})
                length = length + 1
                tempy = tempy + 1
            element = {
                        "id": id,
                        "question": text,
                        "position": str(x) + ":" + str(y), #Position der Frage im Grid
                        "slots": slots, #Positionen der Zeichen der Antwort im Grid
                        "length": length
                    }
            JsonArray.append(element)
        elif cell["handle_right_to_bottom"]== True:
            x = cell["x"] + 1
            y = cell["y"]
            text = [e.get("text_value") for e in Puzzle if e.get("x") == x and e.get("y") == y]
            tempx = x - 1
            tempy = y
            length = 0
            found = any(e["x"] == tempx and e["y"] == tempy and e["text"] == False for e in Puzzle)
            slots = []
            while found:
                found = any(e["x"] == cell["x"] and e["y"] == tempy and e["text"] == False for e in Puzzle)
                slots.append({str(tempx)+":"+str(tempy)})
                length = length + 1
                tempy = tempy + 1
            element = {
                        "id": id,
                        "question": text,
                        "position": str(x) + ":" + str(y), #Position der Frage im Grid
                        "slots": slots, #Positionen der Zeichen der Antwort im Grid
                        "length": length
                    }
            JsonArray.append(element)
        elif cell["handle_top_to_right"]== True:
            x = cell["x"]
            y = cell["y"] + 1
            text = [e.get("text_value") for e in Puzzle if e.get("x") == x and e.get("y") == y]
            tempx = x
            tempy = y - 1
            length = 0
            found = any(e["x"] == tempx and e["y"] == tempy and e["text"] == False for e in Puzzle)
            slots = []
            while found:
                found = any(e["x"] == cell["x"] and e["y"] == tempy and e["text"] == False for e in Puzzle)
                slots.append({str(tempx)+":"+str(tempy)})
                length = length + 1
                tempx = tempx + 1
            element = {
                        "id": id,
                        "question": text,
                        "position": str(x) + ":" + str(y), #Position der Frage im Grid
                        "slots": slots, #Positionen der Zeichen der Antwort im Grid
                        "length": length
                    }
            JsonArray.append(element)
        elif cell["double_arrow"] == True:
            #arrow_to_right
            x1 = cell["x"] - 1
            y1 = cell["y"]
            text = [e.get("text_value") for e in Puzzle if e.get("x") == x1 and e.get("y") == y1]
            tempx = x1 + 1
            tempy = y1
            length = 0
            found = any(e["x"] == tempx and e["y"] == tempy and e["text"] == False for e in Puzzle)
            slots = []
            while found:
                found = any(e["x"] == cell["x"] and e["y"] == tempy and e["text"] == False for e in Puzzle)
                slots.append({str(tempx)+":"+str(tempy)})
                length = length + 1
                tempx = tempx + 1
            element = {
                        "id": id,
                        "question": text,
                        "position": str(x1) + ":" + str(y1), #Position der Frage im Grid
                        "slots": slots, #Positionen der Zeichen der Antwort im Grid
                        "length": length
                    }
            JsonArray.append(element)

            #arrow_to_bottom
            x2 = cell["x"]
            y2 = cell["y"] - 1
            text = [e.get("text_value") for e in Puzzle if e.get("x") == x2 and e.get("y") == y2]
            tempx = x2
            tempy = y2 + 1
            length = 0
            found = any(e["x"] == tempx and e["y"] == tempy and e["text"] == False for e in Puzzle)
            slots = []
            while found:
                found = any(e["x"] == cell["x"] and e["y"] == tempy and e["text"] == False for e in Puzzle)
                slots.append({str(tempx)+":"+str(tempy)})
                length = length + 1
                tempy = tempy + 1
            element = {
                        "id": id,
                        "question": text,
                        "position": str(x2) + ":" + str(y2), #Position der Frage im Grid
                        "slots": slots, #Positionen der Zeichen der Antwort im Grid
                        "length": length
                    }
            JsonArray.append(element)
    return JsonArray