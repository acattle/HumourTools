'''
Created on Nov 22, 2016

@author: Andrew
'''

from igraph import Graph
from igraph.drawing import plot
from math import log, exp


pajekFile = "Data/PairsFSG2.net"
# pajekFile = "Data/pajek_subset.net"

graph = Graph.Read_Pajek(pajekFile)
graph.vs["name"] = graph.vs["id"] #work around for bug: https://github.com/igraph/python-igraph/issues/86
words = set(graph.vs["name"])

#Set the weights to the -log of their forward strength so we can use built-in Dijkstra
logWeights = []
for weight in graph.es["weight"]:
    logWeights.append(-log(weight))
graph.es["p weight"] = graph.es["weight"]
graph.es["weight"] = logWeights

# sail_path_f =  graph.get_shortest_paths("SAIL", "OLYMPICS", weights=graph.es["weight"], output="vpath") #returns path
# sail_path_b =  graph.get_shortest_paths("OLYMPICS", "SAIL", weights=graph.es["weight"], output="vpath") #returns path
# sail_subvertices = set()
# for vertex in sail_path_f[0]:
#     sail_subvertices.update(graph.neighbors(vertex, mode="ALL"))
# for vertex in sail_path_b[0]:
#     sail_subvertices.update(graph.neighbors(vertex, mode="ALL"))
# #         subvs = self.graph.vs.select(subvertices)
# #         subgraph = self.graph.subgraph(subvs)
# sail_subgraph = graph.induced_subgraph(sail_subvertices)
# sail_subgraph.vs["label"] = sail_subgraph.vs["name"]
# 
# sail_subgraph.delete_vertices(["ENGINE", "KITE", "WRECK", "HURRICANE", "DRIVEWAY", "FALL", "SPY", "STICKER", "FLUID", "HITCH", "TAG", "WRENCH", "PUFF", "INTERSTATE", "COMPACT", "BLOCKADE", "VAN", "CHICAGO", "FIX", "DUST", "TWISTER", "LEASE", "MAP", "MONEY", "PORCH", "HONK", "WATER", "BREEZE", "BUMPER", "CANAL", "VENT", "USED", "PATROL", "RINGS", "SEAT", "AHOY", "JEEP", "COMPANY", "ROPE", "TRUNK", "PEDAL", "PIRATE", "FLAP", "DRIVER", "HAZARD", "AMBULANCE", "ACTIVE", "CLARINET", "COOL", "GULLY", "RECALL", "CONTROLS", "BREATH", "TRAFFIC", "IMPULSE", "CANVASS", "EXCEL", "PICK UP", "HAIR", "STARBOARD", "SATURN", "ENGINEER", "BIKE", "TRUMPET", "ANCHOR", "OUT", "HOSE", "BALLOON", "JUMP", "SOFTBALL", "STOPLIGHT", "TRIUMPH", "VOYAGE", "INSPECT", "COUGAR", "FISHING", "LEAK", "TRUCK", "CHASE", "CHAUFFEUR", "TOY", "DECK", "TOW", "BAY", "TON", "HOBBY", "PEER", "CAB", "SHINY", "HIT", "COAT", "POLICEMAN", "VACATION", "NOSE", "CROCHET", "CHIME", "CONQUEST", "GREASE", "CONSOLE", "CABLE", "KEYS", "RAIN", "DOCK", "SALESMAN", "TUNE", "LEAGUE", "PRODUCT", "SIGNAL", "TOYOTA", "SALES", "STERN", "TAXI", "WAGON", "DASHBOARD", "AIRPLANE", "REEL", "ECONOMY", "PLUG", "RIVER", "BICYCLE", "BOTTOM", "PRO", "JOB", "AEROBICS", "NAVIGATOR", "ROAD", "ATHLETIC", "PORSCHE", "CURVE", "PISTON", "POLISH", "TICKET", "PIER", "MANUFACTURER", "UNLOAD", "DAMAGE", "LUBRICATE", "TRAIN", "HOUSE", "PLATES", "INJURY", "TRANSPORTATION", "HULL", "BOWLING", "MONARCH", "COLD", "WINDSHIELD", "TERRIFIC", "DENT", "SUBWAY", "HINDSIGHT", "VELOCITY", "PHYSICAL", "BARRACUDA", "RUST", "VIKING", "JACK", "COMPOUND", "ADJUSTMENT", "HIGHWAY", "JOCK", "CAPTAIN", "DATSUN", "TUBA", "CREW", "INSURANCE", "STEEL", "GONE", "HORN", "STALL", "CURB", "TEAM", "HATCH", "SQUEAK", "STEER", "SQUEAL", "LANE", "TRADE", "SEA", "GAUGE", "SWAY", "HANDLE", "BUMP", "CART", "PEDESTRIAN", "GLIDE", "CANDLE", "EXPENSE", "TRIP", "CHANNEL", "PADDY", "CLAMP", "TORNADO", "WHISTLE", "KEY", "SPEED", "TUG", "BRIDGE", "BOMB", "ANTENNA", "SHIFT", "CONTACT", "SINK", "SHUTTLE", "SEATBELT", "IMPACT", "GRAVY", "BASKETBALL", "AUTOMOBILE", "RELIABLE", "DRIFT", "POP", "RACE", "PERMIT", "VEHICLE", "BANDANNA", "YIELD", "MACHINE", "THING", "STORM", "MILL", "POOL", "SUBMARINE", "FLOAT", "KLEENEX", "CARRIAGE", "CARELESS", "ADVERTISEMENT", "PARKING", "FUEL", "POSSESSION", "BOX", "GEAR", "STREET", "UPSTREAM", "STEAM", "ACCIDENT", "CRASH", "DEAL", "TENNIS", "BLAST", "RIM", "THEFT", "ACTIVITY", "EXPLORER", "DASH", "PASS", "COACH", "OIL", "KIT", "NEUTRAL", "BUYER", "ACROBAT", "CHAMPION", "LIGHT", "FREEWAY", "SUCK", "PORT", "BUY", "BUS", "RIDE", "BUMPS", "BUMPY", "Q-TIPS", "LIBERTY", "FOOT", "BREEZEWAY", "HALL", "TRAILER", "DEMOLISH", "BUDGET", "DRYER", "STRIKE", "POSSESS", "COP", "DOWNSTREAM", "TITLE", "FISH", "HARD", "VESSEL", "TECHNICIAN", "LEMON", "SHRIMP", "CARAVAN", "TRAUMA", "MOTOR", "ARRIVE", "SIREN", "COVER", "FACTORY", "FREE", "STANDARD", "FUSE", "WAX", "CABOOSE", "TRAVEL", "OWNER", "AIR", "WASH", "LOT", "LEGEND", "HOOD", "POLLUTION", "CONTROL", "SNEAKER", "OWN", "POLICE", "AWAY", "GAS", "FUMES", "BOXER", "INTEREST", "HORNET", "AUTO", "FOOTBALL", "PETROLEUM", "CRICKET", "BARCELONA", "ERUPT", "CYCLONE", "ESTIMATE", "RIDER", "WHEEL", "PASSENGER", "BATTERY", "DETAIL", "ELEVATOR", "PLANE", "IDOL", "DRY", "REPAIR", "LICENSE", "MOVE", "PARK", "PART", "RAILROAD", "TROPHY", "CUSTOM", "TURN", "LUXURY", "MECHANICAL", "COCAINE", "EXPENSIVE", "VALVE", "SHIELD", "COMPASS", "EXPLODE", "NAVY", "PURCHASE", "STEREO", "CHALLENGING", "MONOPOLY", "HARBOR", "MAIN", "PAYMENT", "DIAMETER", "CYLINDER", "HANDKERCHIEF", "CHALLENGE", "BURN", "CHOKE", "LAKE", "HITCHHIKE", "FLASHLIGHT", "WHISPER", "ROD", "SELLER", "MOTION", "BRAKE", "AMP", "BUBBLE", "BOULEVARD", "RADIATOR", "OCEAN", "CRUISE", "TROMBONE", "SCARF", "LAUNCH", "DRAFT", "UP", "CAUTION", "SMASH", "SERVICE", "DEER", "MAROON", "MOTORCYCLE", "NEW", "MODEL", "BUGLE", "INFINITY", "DEFROST", "SALE", "CRAFT", "TYPHOON", "EARTH", "RACQUETBALL", "ASHTRAY", "YACHT", "DRAG", "KILOMETER", "LIMOUSINE", "SPORTSMAN", "BUGGY", "WINDING", "MOAT", "RECKLESS", "JUNK", "GOLF", "SQUAD", "MARINE", "VEER", "LOTS", "FLUTE", "BRAVADO", "PROFESSIONAL", "SINKER", "BARTER", "HIKER", "START", "WAKE", "ACCELERATE", "CANVAS", "STOLEN", "AMATEUR", "RACQUET", "WEST", "SNOW", "GARAGE", "MECHANIC", "DROVE", "CHILL",\
#                                "PRACTICE", "DRIVE", "WINNERS", "BASEBALL", "FAST", "TIRE", "CANOE", "BOXING", "OAR", "PERSPIRE", "FAN", "HOCKEY", "SOCCER", "ROW", "BALL", "PLAYER", "HOT", "RAFT", "PERFORMANCE", "SHIP", "WRESTLING", "ICE SKATING", "PADDLE", "SPORTS", "SAILOR", "COMPETE", "MUSCLE", "EVENT",\
#                                "SURF", "SWIMMER", "HEALTHY", "GAMES", "STRONG", "ABILITY", "SHOW", "COMPETITION", "FIRE", "EXERCISE", "SAILING", "CONTEST", "GYMNAST", "GYMNASTICS", "SWEAT", "RUNNER", "FLAME", "BRONZE", "SWIMMING",\
#                                "SCORE", "BOARD", "RUN", "VOLLEYBALL", "GAME", "MEDAL", "GOLD"
#                                ])
# 
# 
# 
# # sail_subgraph.delete_vertices(["NOSE", "AWAY", "HEDGE", "HIT", "RAIN", "OUT", "JOB", "COLUMN", "COCAINE", "HOUSE", "FREE", "PLUG", "EARTH", "CURVE", "BREATH", "HORN", "CHAIR", "HAIR", "DRYER", "SHIELD", "CRAFT", "BOMB", "EXPLODE", "CAUTION", "EGGS", "FISH", "SHRIMP", "POP", "CLARINET", "FLUTE", "SCARF", "HANDKERCHIEF", "PIRATE", "BARTER", "TRADE", "BOTTOM", "BUBBLE", "STRIKE", "LIBERTY", "CANDLE", "GRAVY", "BLAST", "SUCK", "TRAILER", "STEAM", "STERN", "BREEZEWAY", "BUGLE", "CANVASS", "CHIME", "VIKING", "TRAUMA", "CYCLONE", "TWISTER", "GULLY", "LEAK", "DUCKS", "ERUPT", "MILL", "REEL", "FLAP", "TROMBONE", "KITE", "FUSE", "AISLE", "KLEENEX", "HATCH", "HITCH", "WHISPER", "TUBA", "TUG", "LAUNCH", "PADDY", "STARBOARD",\
# #                           "WORK", "UP", "GONE", "WATER", "CAR", "FALL", "BURN", "TRIP", "AIR", "HOT", "FRONT", "AHOY", "SHIP", "CAPTAIN", "FIRE", "PLANE", "BALLOON", "NAVY", "AIRPLANE", "LAKE", "OCEAN", "HARD", "WAKE", "COLD", "DUST", "RIVER", "LIGHT", "ANCHOR", "SEA", "LINE", "SNOW", "COOL", "MARINE", "CANVAS", "SINK", "FLASHLIGHT", "TOY", "BAY", "HARBOR", "DRY", "WHISTLE", "MOTOR", "FISHING", "TRUMPET", "BREEZE", "WEST", "STORM", "CANAL", "DECK", "CRUISE", "CREW", "MOAT", "HURRICANE", "TORNADO", "FAN", "MAIN", "CHANNEL", "CHILL", "TOW", "GLIDE", "PEER", "COMPASS", "WINDING", "DOCK", "PIER", "PORT", "DOWNSTREAM", "UPSTREAM", "DRAFT", "PUFF", "DRIFT", "SWAY", "NAVIGATOR", "SUBMARINE", "SINKER", "HULL", "TYPHOON", "VOYAGE",\
# #                           "ROPE", "POOL", "WRESTLING", "FLOAT", "SURF", "BOARD", "RINGS", "FLAME", "VESSEL", "YACHT", "CANOE", "CHICAGO", "RAFT", "VELOCITY", "BARCELONA",
# #                           "JUMP", "WINNERS", "CHALLENGING", "ICE SKATING", "FIGHT", "SAILOR", "RUN", "EVENT", "GAMES", "ATHLETE", "GYMNASTICS", "SHOW", "SWIMMING", "ACROBAT"
# #  
# #                            
# #                           ])
# 
#     
# #https://stackoverflow.com/questions/10067721/can-i-change-the-colour-of-edges-containing-specific-vertices-in-igraph-python
# sail_subgraph.es["color"] = "black"
# 
# sail_path_fs =  sail_subgraph.get_shortest_paths("SAIL", "OLYMPICS", weights=sail_subgraph.es["weight"], output="vpath") #returns path
# sail_path_fs_strength = sail_subgraph.shortest_paths("SAIL", "OLYMPICS", weights=sail_subgraph.es["weight"])[0][0] #returns value
# for i in range(len(sail_path_fs[0])-1):
#     e = sail_subgraph.es.select(_from=sail_path_fs[0][i], _to=sail_path_fs[0][i+1])
#     e["color"] = "#C05A4D"
#     e["width"] = 4
# sail_path_bs =  sail_subgraph.get_shortest_paths("OLYMPICS", "SAIL", weights=sail_subgraph.es["weight"], output="vpath") #returns path
# sail_path_bs_strength = sail_subgraph.shortest_paths("OLYMPICS", "SAIL", weights=sail_subgraph.es["weight"])[0][0] #returns value
# for i in range(len(sail_path_bs[0])-1):
#     e=sail_subgraph.es.select(_from=sail_path_bs[0][i], _to=sail_path_bs[0][i+1])
#     e["color"] = "#92D050"
#     e["width"] = 4
# 
# #TODO: I got forward and backward mixed up
# print("FWAforward(OLYMPICS, SAIL) = {}".format(exp(-sail_path_fs_strength)))
# print("FWAbackward(OLYMPICS, SAIL) = {}".format(exp(-sail_path_bs_strength)))
# 
# sail_subgraph.vs["color"] = "#1F497D"
# # sail_subgraph.vs["color"] = ["#1F497D" for vertex in sail_subgraph.vs]
# sail_subgraph.vs.find(name="SAIL")["color"] = "#C05A4D"
# sail_subgraph.vs.find(name="OLYMPICS")["color"] = "#92D050"
# 
# sail_subgraph.vs["label_size"] = [14 for vertex in sail_subgraph.vs]
# # sail_subgraph.vs.find(name="COMPETITION")["label_size"] = 12
# # sail_subgraph.vs.find(name="VOLLEYBALL")["label_size"] = 13
# 
# sail_subgraph.es["label"] = sail_subgraph.es["p weight"] #https://stackoverflow.com/questions/21140853/labelling-the-edges-in-a-graph-with-python-igraph
# 
# #http://www.cs.rhul.ac.uk/home/tamas/development/igraph/tutorial/tutorial.html#drawing-a-graph-using-a-layout
# layout = sail_subgraph.layout("kk")
visual_style = {}
visual_style["vertex_size"] = 20
# visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
# visual_style["vertex_label"] = g.vs["name"]
# visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
visual_style["vertex_size"] = 75
visual_style["vertex_label_color"] = "white"
visual_style["bbox"] = (1600, 900)
visual_style["margin"] = 100
visual_style["edge_label_size"] = 13
# plot(sail_subgraph, r"C:\users\Andrew\Desktop\coling paper\poster\sail_fwa_test.png", layout=layout, **visual_style)
# 
# 
# 
# row_path_f =  graph.get_shortest_paths("ROW", "OLYMPICS", weights=graph.es["weight"], output="vpath") #returns path
# row_path_b =  graph.get_shortest_paths("OLYMPICS", "ROW", weights=graph.es["weight"], output="vpath") #returns path
# row_subvertices = set()
# for vertex in row_path_f[0]:
#     row_subvertices.update(graph.neighbors(vertex, mode="ALL"))
# for vertex in row_path_b[0]:
#     row_subvertices.update(graph.neighbors(vertex, mode="ALL"))
#     
# row_subgraph = graph.induced_subgraph(row_subvertices)
# row_subgraph.vs["label"] = row_subgraph.vs["name"]
# 
# row_subgraph.delete_vertices(["DRY", "PUNISHMENT", "BACKSTROKE", "GEAR", "GONE", "HURRICANE", "NAVY", "UPSTREAM", "FLIPPER", "AIRPLANE", "REEL", "STEAM", "KITE", "PLUG", "TABLE", "TWISTER", "BOTTOM", "FIN", "SHALLOW", "HITCH", "LOUNGE", "JOB", "AIR", "KIDNEY", "SUIT", "FRONT", "PUFF", "HEDGE", "COCAINE", "MEDAL", "HOBBY", "CHIME", "SHIELD", "CHICAGO", "EGGS", "CURVE", "FLUTE", "EXPLODE", "TRIP", "WET", "DUST", "PARTY", "RIVER", "PIER", "CANVASS", "HAIR", "WHIP", "CHALLENGING", "HARBOR", "HOUSE", "LAKE", "SPANK", "WATER", "BREEZE", "COMPETITION", "CAPTAIN", "LAP", "ACROBAT", "MAN", "FLY", "CHAMPION", "HANDKERCHIEF", "LIGHT", "FALL", "RINGS", "FIGHT", "GAMES", "AHOY", "COLD", "LINE", "MERMAID", "FLASHLIGHT", "UNDERWATER", "SHARK", "WHISTLE", "BEACH", "DOLPHIN", "ROPE", "MEET", "CHAIR", "BUBBLE", "PIRATE", "VIKING", "FLAP", "CANAL", "LIBERTY", "OCEAN", "STROKE", "RAIN", "CRUISE", "VESSEL", "CLARINET", "BREEZEWAY", "HALL", "SCARF", "COOL", "HOT", "MAIN", "GULLY", "LAUNCH", "TUBE", "ICE SKATING", "TUBA", "FLUTTER", "BREATH", "COLUMN", "PUMP", "DRYER", "CAUTION", "STRIKE", "FLAME", "GYMNASTICS", "ALGAE", "STAFF", "SUMMER", "DOWNSTREAM", "SPOOL", "FISH", "HARD", "TREAD", "HORN", "HURT", "TROMBONE", "WINNERS", "COMPETE", "HATCH", "GYMNAST", "STARBOARD", "OVERFLOW", "SHRIMP", "BUGLE", "NAVIGATOR", "GOLD", "SWAM", "WORK", "FAST", "TRADE", "SMACK", "DROWN", "TYPHOON", "TRAILER", "HULL", "TRAUMA", "TRUMPET", "OUT", "DUNK", "BURN", "MOTOR", "BALLOON", "FREE", "JUMP", "SUBMARINE", "CUE", "FUSE", "EVENT", "EARTH", "CANDLE", "BAY", "CHILDHOOD", "CHANNEL", "PADDY", "BLUE", "VOYAGE", "CRAMP", "CREEK", "TORNADO", "SWAY", "WRESTLING", "WINDING", "FISHING", "LEAK", "MOAT", "SIDE", "TUG", "AISLE", "TOY", "BOMB", "GLIDE", "DECK", "VELOCITY", "CAR", "TOW", "UP", "DEPTH", "BALL", "DRAFT", "PEER", "CREW", "MARINE", "SUCK", "PATIO", "GRAVY", "HIT", "REFLECTION", "AWAY", "CAMP", "DRIFT", "POP", "PORT", "BRONZE", "DUCKS", "FAN", "NOSE", "COMPASS", "CRAFT", "RACK", "SARDINE", "LAGOON", "BARTER", "FIRE", "STERN", "BARCELONA", "POCKET", "DOCK", "HEAT", "ERUPT", "WAKE", "STORM", "CYCLONE", "MILL", "CANVAS", "BLAST", "DIP", "WEST", "SNOW", "VOLLEYBALL", "KLEENEX", "PLANE", "WHISPER", "GENE", "CHILL",\
#                               "SAILOR", "SINK", "DUCK", "POND", "SCUBA", "CHLORINE", "CANOE", "SNORKEL", "GYM", "SPLASH", "SEA", "DIVER", "EXERCISE", "RUNNER", "DIVE", "YACHT", "DIVING", "RUN", "LAPS", "FLOAT", "ANCHOR", "SINKER", "RAFT",\
#                               "SHOW", "EXCITING", "FUN", "SKI", "BOARD", "OAR", "SWIMMING", "SAILING"
#                               ])
# 
# row_subgraph.es["color"] = "black"
# 
# 
# 
# row_path_fs =  row_subgraph.get_shortest_paths("ROW", "OLYMPICS", weights=row_subgraph.es["weight"], output="vpath") #returns path
# row_path_fs_strength = row_subgraph.shortest_paths("ROW", "OLYMPICS", weights=row_subgraph.es["weight"])[0][0] #returns value
# for i in range(len(row_path_fs[0])-1):
#     e=row_subgraph.es.select(_from=row_path_fs[0][i], _to=row_path_fs[0][i+1])
#     e["color"] = "#C05A4D"
#     e["width"] = 4
# row_path_bs =  row_subgraph.get_shortest_paths("OLYMPICS", "ROW", weights=row_subgraph.es["weight"], output="vpath") #returns path
# row_path_bs_strength = row_subgraph.shortest_paths("OLYMPICS", "ROW", weights=row_subgraph.es["weight"])[0][0] #returns value
# for i in range(len(row_path_bs[0])-1):
#     e=row_subgraph.es.select(_from=row_path_bs[0][i], _to=row_path_bs[0][i+1])
#     e["color"] = "#92D050"
#     e["width"] = 4
# 
# #TODO: I got forward and backward mixed up
# print("FWAforward(OLYMPICS, ROW) = {}".format(exp(-row_path_fs_strength)))
# print("FWAbackward(OLYMPICS, ROW) = {}".format(exp(-row_path_bs_strength)))
# 
# path_nodes = []
# for i in range(len(row_path_b[0])):
# #     print(graph.vs[row_path_b[0][i]]["name"])
#     path_nodes.append(graph.vs[row_path_b[0][i]]["name"])
#     if graph.vs[row_path_b[0][i]]["name"] != row_subgraph.vs[row_path_bs[0][i]]["name"]:
#         print("ERROR!")
# for i in range(len(row_path_f[0])):
# #     print(graph.vs[row_path_f[0][i]]["name"])
#     path_nodes.append(graph.vs[row_path_f[0][i]]["name"])
#     if graph.vs[row_path_f[0][i]]["name"] != row_subgraph.vs[row_path_fs[0][i]]["name"]:
#         print("ERROR!")
# 
# print("\n\n")
# # path_nodes = set(path_nodes)
# # nodes = set(row_subgraph.vs["name"])
# # for word in (nodes - path_nodes):
# #     print(word)
#     
# 
# row_subgraph.vs["color"] = "#1F497D"
# row_subgraph.vs.find(name="ROW")["color"] = "#C05A4D"
# row_subgraph.vs.find(name="OLYMPICS")["color"] = "#92D050"
# 
# row_subgraph.vs["label_size"] = [14 for vertex in row_subgraph.vs]
# # sail_subgraph.vs.find(name="COMPETITION")["label_size"] = 12
# # sail_subgraph.vs.find(name="VOLLEYBALL")["label_size"] = 13
# 
# row_subgraph.es["label"] = row_subgraph.es["p weight"] #https://stackoverflow.com/questions/21140853/labelling-the-edges-in-a-graph-with-python-igraph
# 
# #http://www.cs.rhul.ac.uk/home/tamas/development/igraph/tutorial/tutorial.html#drawing-a-graph-using-a-layout
# layout = row_subgraph.layout("kk")
# plot(row_subgraph, r"C:\users\Andrew\Desktop\coling paper\poster\row_fwa_test.png", layout=layout, **visual_style)
# print("done")

olympics_subvertices = set(["OLYMPICS"])
olympics_subvertices.update(graph.neighbors("OLYMPICS", mode="ALL"))
olympics = graph.induced_subgraph(olympics_subvertices)
olympics.vs["label_size"] = [14 for vertex in olympics.vs]
olympics.es["label"] = olympics.es["p weight"]
olympics.vs["label"] = olympics.vs["name"]
olympics.delete_vertices(["EXCITING", "CHALLENGING","BARCELONA", "GYMNASTICS", "WRESTLING", "ICE SKATING","COMPETITION","COMPETE","SWIMMER",
                          "VOLLEYBALL","RUNNER", "ACROBAT", "POOL", "GYMNAST", "EVENT", "WINNERS", "SWIMMING", "SUMMER", "RINGS"
                            ])
layout=olympics.layout("kk")
plot(olympics, r"C:\users\Andrew\Desktop\olympics.png", layout=layout, **visual_style)