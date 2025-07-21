# README

### Install Dependencies
```
pip install openai
```


### Markup Syntax
#### Chat History File
save in `./history`
```text
# Use this tag to distinguish between different roles.
========== @{role} ==========  

========== @system ==========  # System prompt

========== @user ==========  # User's questions

========== @assistant ==========  # AI assistant's answers
```

#### Multi-language Prompts File
save in `./prompts`
```text
# Use this tag to distinguish between different translation.
========== #{language} ==========
```