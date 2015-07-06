using Tk,Winston,Graphics

function stats(M::EconModel.Model)
	Top = Toplevel("Model Summary", 800, 600)
	Fmain = Frame(Top)
	grid_columnconfigure(Fmain, 1, weight=1)
	grid_rowconfigure(Fmain, 1, weight=1)
	pack(Fmain, expand=true, fill="both")
	Ftitle = Label(Fmain,"label")
	Tk.grid(Ftitle, 1, 1:2, sticky="nsew")
	c = Canvas(Fmain, 750, 600)
	Tk.grid(c, 2:25, 1, sticky="nsew")

	Fcontrols = Frame(Fmain)
	Tk.grid(Fcontrols, 2:25, 2, sticky="nw")


	controls = (Radio(Fcontrols,["Error";"Variables"]),
				Combobox(Fcontrols,String[string(x) for x in M.state.names]),
				Combobox(Fcontrols,String[string(x) for x in M.state.names]),
				Combobox(Fcontrols,String[string(x) for x in 1:M.policy.n]),
				Combobox(Fcontrols,vcat(String[string(x) for x in M.policy.names],String[string(x) for x in M.static.names])),
				Button(Fcontrols, "Update"))

labels = (Label(Fcontrols,"x"),
			  Label(Fcontrols,"colour"),
			  Label(Fcontrols,"error "),
			  Label(Fcontrols,"var"))


	Tk.grid(controls[1], 1, 1:10)
    grid_configure(controls[1], Dict(:sticky => "we"))
    Tk.grid(controls[end], length(controls), 1:10)
    grid_configure(controls[end], Dict(:sticky => "we"))

	for i in 2:length(controls)-1
        Tk.grid(controls[i], i, 3:10)
        Tk.grid(labels[i-1],i,1:2)
        grid_configure(controls[i], Dict(:sticky => "we"))
    end

	

    function updateplot(path)
    	eplot = get_value(controls[1])=="Error"
		iv = findfirst(symbol(get_value(controls[2])).==M.state.names)
		ic = findfirst(symbol(get_value(controls[3])).==M.state.names)
		ie = parse(Int,get_value(controls[4]))
		
		set_value(Ftitle,string(M.meta.foc.args[ie]))
		if eplot
			p = Winston.scatter(M.state.G.grid[:,iv],M.error[:,ie],.1,M.state.G.grid[:,ic])
		else
			V = symbol(get_value(controls[5]))
			if V in M.policy.names
				iV = findfirst(V.==M.policy.names)
				p = Winston.scatter(M.state.G.grid[:,iv],M.policy.X[:,iV],.1,M.state.G.grid[:,ic])
			elseif V in M.static.names
				iV = findfirst(V.==M.static.names)
				p = Winston.scatter(M.state.G.grid[:,iv],M.static.X[:,iV],.1,M.state.G.grid[:,ic])
			end
		end
		Winston.display(c, p)
		reveal(c)
		Tk.update()
	end


	bind(controls[end],"command",updateplot)
	p = Winston.scatter(M.state.G.grid[:,1],M.error[:,1],.1);
	

	Winston.display(c, p);
	reveal(c)
	Tk.update()

end
