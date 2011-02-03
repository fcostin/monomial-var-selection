rule '.extendedcsv' => ['.csv', 'add_monomials.py'] do |t|
	sh "python add_monomials.py --source #{t.source} --dest #{t.name} --max_degree #{2}"
end
