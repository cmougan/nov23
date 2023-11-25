black:
	python -m black .
gitall:
	git add .
	@read -p "Enter commit message: " message; 	git commit -m "$$message"
	git push

update_branch:
	git checkout main
	git pull
	git checkout carlos
	git fetch
	git rebase origin/main